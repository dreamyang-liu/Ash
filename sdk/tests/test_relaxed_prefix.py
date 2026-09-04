from ash_sandbox.relaxed_prefix import (
    BARRIER,
    MUTATION,
    SAFE_READ,
    classify_tool_effect,
    normalize_tool_result_messages,
    project_environment_prefix,
    read_result_cache_key,
    is_proven_workspace_read_shell,
)


def tr(tool, args, content="ok", success=True):
    return {
        "role": "tool_result",
        "tool_name": tool,
        "tool_args": args,
        "content": content,
        "success": success,
    }


def test_p0_effect_classifier_is_conservative():
    assert classify_tool_effect("grep_files", {"pattern": "x"}) == SAFE_READ
    assert classify_tool_effect("text_editor", {"command": "view"}) == SAFE_READ
    assert classify_tool_effect("text_editor", {"command": "str_replace"}) == MUTATION
    assert classify_tool_effect("shell", {"command": "cat a.py"}) == BARRIER
    assert classify_tool_effect("process", {"action": "read"}) == BARRIER


def test_safe_reads_do_not_change_relaxed_environment_state_key():
    a = [
        tr("grep_files", {"pattern": "foo", "path": "/testbed"}, "a.py:1"),
        tr("text_editor", {"command": "view", "path": "/testbed/a.py"}, "content"),
    ]
    b = list(reversed(a))
    pa = project_environment_prefix(a)
    pb = project_environment_prefix(b)
    assert pa.state_hash == pb.state_hash
    assert pa.state_steps == pb.state_steps == 0
    assert pa.ignored_read_steps == pb.ignored_read_steps == 2
    assert pa.model_prefix_reusable is False
    assert pa.kv_reuse is False


def test_mutation_order_is_never_relaxed():
    edit_a = tr("text_editor", {"command": "str_replace", "path": "/testbed/a.py", "old_str": "a", "new_str": "b"}, "done-a")
    edit_b = tr("text_editor", {"command": "str_replace", "path": "/testbed/b.py", "old_str": "x", "new_str": "y"}, "done-b")
    assert project_environment_prefix([edit_a, edit_b]).state_hash != project_environment_prefix([edit_b, edit_a]).state_hash


def test_unknown_shell_is_a_state_barrier_even_when_command_looks_read_only():
    grep = tr("grep_files", {"pattern": "foo"}, "foo")
    shell = tr("shell", {"command": "cat a.py"}, "abc")
    p = project_environment_prefix([grep, shell])
    assert p.tool_steps == 2
    assert p.ignored_read_steps == 1
    assert p.state_steps == 1


def test_different_mutation_outcomes_get_different_state_keys():
    args = {"command": "str_replace", "path": "/testbed/a.py", "old_str": "a", "new_str": "b"}
    ok = project_environment_prefix([tr("text_editor", args, "replaced", True)])
    fail = project_environment_prefix([tr("text_editor", args, "old_str not found", False)])
    assert ok.state_hash != fail.state_hash


def test_read_result_cache_key_is_scoped_to_environment_state():
    root = project_environment_prefix([]).state_hash
    mutated = project_environment_prefix([tr("text_editor", {"command": "write", "path": "/testbed/a.py", "file_text": "x"}, "done")]).state_hash
    args = {"pattern": "TODO", "path": "/testbed"}
    k1 = read_result_cache_key(root, "grep_files", args)
    assert k1 == read_result_cache_key(root, "grep_files", args)
    assert k1 != read_result_cache_key(mutated, "grep_files", args)


def test_read_result_cache_rejects_unproven_shell_reads():
    state = project_environment_prefix([]).state_hash
    try:
        read_result_cache_key(state, "shell", {"command": "cat a.py"})
    except ValueError as exc:
        assert "proven safe reads" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_p1_shell_allowlist_accepts_only_path_scoped_simple_reads():
    assert is_proven_workspace_read_shell("cd /testbed && ls")
    assert is_proven_workspace_read_shell("cd /testbed && cat django/conf/global_settings.py")
    assert is_proven_workspace_read_shell("cat /testbed/django/conf/global_settings.py")
    assert is_proven_workspace_read_shell("cd /testbed && grep -n TODO django/conf/global_settings.py")
    assert not is_proven_workspace_read_shell("cat /etc/passwd")
    assert not is_proven_workspace_read_shell("cd /testbed && cat a.py > /tmp/out")
    assert not is_proven_workspace_read_shell("cd /testbed && git status")
    assert not is_proven_workspace_read_shell("cd /testbed && find . -delete")
    assert not is_proven_workspace_read_shell("cd /testbed && tail -f logs.txt")


def test_p1_shell_read_grammar_accepts_safe_composition_and_rejects_mutation():
    assert is_proven_workspace_read_shell(
        "cd /testbed; grep -n TODO . 2>/dev/null | sort | uniq"
    )
    assert is_proven_workspace_read_shell("sed -n '1,120p' /testbed/a.py")
    assert is_proven_workspace_read_shell("find /testbed -maxdepth 2 -type f | head -20")
    assert is_proven_workspace_read_shell(
        "cat /testbed/a.py 2>/dev/null; head -2 /testbed/a.py"
    )
    assert not is_proven_workspace_read_shell("sed -i 's/a/b/' /testbed/a.py")
    assert not is_proven_workspace_read_shell("find /testbed -type f -delete")
    assert not is_proven_workspace_read_shell("cd /testbed; cat a.py | tee copy.py")
    assert not is_proven_workspace_read_shell("echo $(touch /testbed/pwned)")


def test_p1_shell_read_grammar_requires_explicit_trusted_external_roots():
    command = "sed -n '1,120p' /usr/local/cargo/git/checkouts/pkg/src/lib.rs"
    assert not is_proven_workspace_read_shell(command, workspace_roots=("/app",))
    assert is_proven_workspace_read_shell(
        command,
        workspace_roots=("/app", "/usr/local/cargo"),
    )


def test_p1_shell_read_grammar_uses_explicit_trusted_working_directory():
    assert is_proven_workspace_read_shell(
        "find src -type f | sort; rg -n TODO src",
        workspace_roots=("/app",),
        working_dir="/app",
    )
    assert not is_proven_workspace_read_shell(
        "find src -type f | sort",
        workspace_roots=("/app",),
        working_dir="/tmp",
    )


def test_p1_classifier_forwards_shell_working_directory():
    assert classify_tool_effect(
        "shell",
        {"command": "cat README.md", "working_dir": "/app"},
        allow_safe_shell=True,
        workspace_roots=("/app",),
    ) == SAFE_READ


def test_p1_projection_can_ignore_proven_workspace_shell_reads():
    shell_read = tr("shell", {"command": "cd /testbed && cat django/conf/global_settings.py"}, "contents")
    p0 = project_environment_prefix([shell_read])
    p1 = project_environment_prefix([shell_read], allow_safe_shell=True)
    assert p0.state_steps == 1 and p0.ignored_read_steps == 0
    assert p1.state_steps == 0 and p1.ignored_read_steps == 1
    key = read_result_cache_key(
        p1.state_hash, "shell", {"command": "cd /testbed && cat django/conf/global_settings.py"}, allow_safe_shell=True
    )
    assert len(key) == 64


def test_model_facing_openai_messages_are_reconstructed_into_tool_results():
    messages = [
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "text_editor",
                        "arguments": '{"command":"view","path":"/testbed/a.py"}',
                    },
                },
                {
                    "id": "call-2",
                    "type": "function",
                    "function": {
                        "name": "text_editor",
                        "arguments": '{"command":"str_replace","path":"/testbed/a.py","old_str":"a","new_str":"b"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "contents"},
        {"role": "tool", "tool_call_id": "call-2", "content": "replaced"},
    ]
    normalized = list(normalize_tool_result_messages(messages))
    assert [m["tool_name"] for m in normalized] == ["text_editor", "text_editor"]
    assert normalized[0]["tool_args"] == {"command": "view", "path": "/testbed/a.py"}
    assert normalized[1]["tool_args"]["command"] == "str_replace"

    projection = project_environment_prefix(messages)
    assert projection.tool_steps == 2
    assert projection.ignored_read_steps == 1
    assert projection.state_steps == 1


def test_unpaired_model_tool_message_becomes_conservative_barrier():
    messages = [{"role": "tool", "tool_call_id": "missing", "content": "unexpected"}]
    normalized = list(normalize_tool_result_messages(messages))
    assert normalized[0]["tool_name"] == "__unknown_tool__"
    projection = project_environment_prefix(messages)
    assert projection.state_steps == 1
    assert projection.ignored_read_steps == 0
