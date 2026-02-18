//! ash CLI

use ash::{Tool, ToolResult};
use ash::daemon;
use ash::style;
use ash::tools;
use clap::Parser;
use serde_json::Value;

use std::collections::HashMap;

use ash::cli::*;

/// Execute a tool: route through gateway, fallback to direct execution.
/// Gateway handles all routing: local (via ash-mcp), Docker, K8s.
async fn exec_tool(tool: &dyn Tool, args: Value, session_id: &Option<String>) -> ToolResult {
    // 1. Ensure gateway is running (auto-start if needed)
    daemon::ensure_gateway().await;

    // 2. Route through gateway
    if let Some(result) = daemon::gateway_tool_call(tool.name(), args.clone(), session_id).await {
        return result;
    }

    // 3. Fallback: direct local execution (gateway unavailable)
    if session_id.is_some() {
        return ToolResult::err("Gateway required for session routing but not available".to_string());
    }
    tool.execute(args).await
}

// CLI type definitions are in ash::cli (src/cli.rs)


fn parse_key_value(items: &[String]) -> HashMap<String, String> {
    items.iter()
        .filter_map(|s| {
            let parts: Vec<&str> = s.splitn(2, '=').collect();
            if parts.len() == 2 { Some((parts[0].to_string(), parts[1].to_string())) } else { None }
        })
        .collect()
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let session_id = cli.session.clone();
    
    let result = match cli.command {
        // ==================== File Operations ====================

        Commands::Grep { pattern, path, include, limit } => {
            exec_tool(&tools::GrepTool, serde_json::json!({
                "pattern": pattern, "path": path, "include": include, "limit": limit
            }), &session_id).await
        }

        Commands::Edit { op } => {
            let args = match op {
                EditOp::View { path, start, end } => serde_json::json!({
                    "command": "view", "path": path, "view_range": [start, end]
                }),
                EditOp::Replace { path, old, new } => serde_json::json!({
                    "command": "str_replace", "path": path, "old_str": old, "new_str": new
                }),
                EditOp::Insert { path, line, text } => serde_json::json!({
                    "command": "insert", "path": path, "insert_line": line, "insert_text": text
                }),
                EditOp::Create { path, content } => serde_json::json!({
                    "command": "create", "path": path, "file_text": content
                }),
            };
            exec_tool(&tools::EditTool, args, &session_id).await
        }

        Commands::Find { pattern, path, max_depth, limit } => {
            exec_tool(&tools::FindFilesTool, serde_json::json!({
                "pattern": pattern, "path": path, "max_depth": max_depth, "limit": limit
            }), &session_id).await
        }

        Commands::Tree { path, max_depth, show_hidden } => {
            exec_tool(&tools::TreeTool, serde_json::json!({
                "path": path, "max_depth": max_depth, "show_hidden": show_hidden
            }), &session_id).await
        }

        Commands::Diff { file1, file2, context } => {
            exec_tool(&tools::DiffFilesTool, serde_json::json!({
                "file1": file1, "file2": file2, "context": context
            }), &session_id).await
        }

        Commands::Patch { patch, path, dry_run } => {
            exec_tool(&tools::PatchApplyTool, serde_json::json!({
                "patch": patch, "path": path, "dry_run": dry_run
            }), &session_id).await
        }

        Commands::FileInfo { path } => {
            exec_tool(&tools::FileInfoTool, serde_json::json!({"path": path}), &session_id).await
        }

        Commands::Fetch { url, timeout } => {
            exec_tool(&tools::HttpFetchTool, serde_json::json!({
                "url": url, "timeout_secs": timeout
            }), &session_id).await
        }

        Commands::Undo { path, list } => {
            exec_tool(&tools::UndoTool, serde_json::json!({"path": path, "list": list}), &session_id).await
        }

        Commands::Outline { file_path } => {
            exec_tool(&tools::OutlineTool, serde_json::json!({"file_path": file_path}), &session_id).await
        }

        // ==================== Filesystem ====================

        Commands::Ls { path } => {
            exec_tool(&tools::FsListDirTool, serde_json::json!({"path": path}), &session_id).await
        }

        // ==================== Buffer ====================

        Commands::Buffer { op } => {
            match op {
                BufferOp::Read { name, start, end } => {
                    exec_tool(&tools::BufferReadTool, serde_json::json!({
                        "name": name, "start_line": start, "end_line": end
                    }), &session_id).await
                }
                BufferOp::Write { name, content, at_line, append } => {
                    exec_tool(&tools::BufferWriteTool, serde_json::json!({
                        "name": name, "content": content, "at_line": at_line, "append": append
                    }), &session_id).await
                }
                BufferOp::Delete { name, start, end } => {
                    exec_tool(&tools::BufferDeleteTool, serde_json::json!({
                        "name": name, "start_line": start, "end_line": end
                    }), &session_id).await
                }
                BufferOp::Replace { name, start, end, content } => {
                    exec_tool(&tools::BufferReplaceTool, serde_json::json!({
                        "name": name, "start_line": start, "end_line": end, "content": content
                    }), &session_id).await
                }
                BufferOp::List => {
                    exec_tool(&tools::BufferListTool, serde_json::json!({}), &session_id).await
                }
                BufferOp::Clear { name } => {
                    exec_tool(&tools::BufferClearTool, serde_json::json!({"name": name}), &session_id).await
                }
            }
        }

        // ==================== Shell ====================

        Commands::Run { command, timeout, tail, revert } => {
            exec_tool(&tools::ShellTool, serde_json::json!({
                "command": command, "timeout_secs": timeout, "tail_lines": tail, "revert_command": revert
            }), &session_id).await
        }

        Commands::RunRevert { id } => {
            exec_tool(&tools::ShellRevertTool, serde_json::json!({"id": id}), &session_id).await
        }

        Commands::RunHistory { limit } => {
            exec_tool(&tools::ShellHistoryTool, serde_json::json!({"limit": limit}), &session_id).await
        }

        Commands::Terminal { op } => {
            let (tool_name, args): (&str, Value) = match op {
                TerminalOp::Start { command, workdir, env, revert } => {
                    let env_map = parse_key_value(&env);
                    ("terminal_run_async", serde_json::json!({
                        "command": command, "working_dir": workdir, "env": env_map, 
                        "revert_command": revert, "session_id": session_id
                    }))
                }
                TerminalOp::Output { handle, tail } => {
                    ("terminal_get_output", serde_json::json!({"handle": handle, "tail": tail, "session_id": session_id}))
                }
                TerminalOp::Kill { handle } => {
                    ("terminal_kill", serde_json::json!({"handle": handle, "session_id": session_id}))
                }
                TerminalOp::List => {
                    ("terminal_list", serde_json::json!({"session_id": session_id}))
                }
                TerminalOp::Remove { handle } => {
                    ("terminal_remove", serde_json::json!({"handle": handle, "session_id": session_id}))
                }
                TerminalOp::Revert { handle } => {
                    ("terminal_revert", serde_json::json!({"handle": handle, "session_id": session_id}))
                }
            };
            // Route through gateway (session_id is embedded in args for terminal tools)
            daemon::ensure_gateway().await;
            if let Some(result) = daemon::gateway_tool_call(tool_name, args.clone(), &None).await {
                result
            } else {
                tools::find_tool(tool_name).unwrap().execute(args).await
            }
        }

        // ==================== Git ====================

        Commands::GitStatus { short } => {
            exec_tool(&tools::GitStatusTool, serde_json::json!({"short": short}), &session_id).await
        }

        Commands::GitDiff { staged, paths } => {
            exec_tool(&tools::GitDiffTool, serde_json::json!({"staged": staged, "paths": paths}), &session_id).await
        }

        Commands::GitLog { count, oneline } => {
            exec_tool(&tools::GitLogTool, serde_json::json!({"count": count, "oneline": oneline}), &session_id).await
        }

        Commands::GitAdd { paths, all } => {
            exec_tool(&tools::GitAddTool, serde_json::json!({"paths": paths, "all": all}), &session_id).await
        }

        Commands::GitCommit { message, all } => {
            exec_tool(&tools::GitCommitTool, serde_json::json!({"message": message, "all": all}), &session_id).await
        }

        // ==================== Session ====================

        Commands::Session { op } => {
            let (tool_name, args): (&str, Value) = match op {
                SessionOp::Create { backend, name, image, port, env, cpu_request, cpu_limit, memory_request, memory_limit, node_selector } => {
                    let env_map = parse_key_value(&env);
                    let node_sel = parse_key_value(&node_selector);

                    // Auto-select docker backend when --image is provided
                    let backend = backend.or_else(|| image.as_ref().map(|_| "docker".to_string()));

                    let mut args = serde_json::json!({
                        "name": name, "image": image, "ports": port, "env": env_map, "node_selector": node_sel,
                        "backend": backend,
                    });

                    let mut resources = serde_json::json!({});
                    if cpu_request.is_some() || memory_request.is_some() {
                        let mut req = serde_json::json!({});
                        if let Some(v) = cpu_request { req["cpu"] = serde_json::json!(v); }
                        if let Some(v) = memory_request { req["memory"] = serde_json::json!(v); }
                        resources["requests"] = req;
                    }
                    if cpu_limit.is_some() || memory_limit.is_some() {
                        let mut lim = serde_json::json!({});
                        if let Some(v) = cpu_limit { lim["cpu"] = serde_json::json!(v); }
                        if let Some(v) = memory_limit { lim["memory"] = serde_json::json!(v); }
                        resources["limits"] = lim;
                    }
                    if !resources.as_object().unwrap().is_empty() { args["resources"] = resources; }

                    ("session_create", args)
                }
                SessionOp::Info { session_id } => {
                    ("session_info", serde_json::json!({"session_id": session_id}))
                }
                SessionOp::List => ("session_list", serde_json::json!({})),
                SessionOp::Destroy { session_id } => {
                    ("session_destroy", serde_json::json!({"session_id": session_id}))
                }
                SessionOp::Switch { session_id, backend } => {
                    ("backend_switch", serde_json::json!({"session_id": session_id, "backend": backend}))
                }
            };
            // Route through gateway (session management runs in gateway process)
            daemon::ensure_gateway().await;
            if let Some(result) = daemon::gateway_tool_call(tool_name, args.clone(), &None).await {
                result
            } else {
                tools::find_tool(tool_name).unwrap().execute(args).await
            }
        }

        // ==================== Config ====================
        
        Commands::Config { control_plane_url, gateway_url } => {
            // Show current backend status instead of old config
            if control_plane_url.is_none() && gateway_url.is_none() {
                tools::BackendStatusTool.execute(serde_json::json!({})).await
            } else {
                // Configure K8s backend with new URLs
                use ash::backend::K8sConfig;
                let config = K8sConfig {
                    control_plane_url: control_plane_url.unwrap_or_else(|| {
                        std::env::var("ASH_CONTROL_PLANE_URL").unwrap_or_else(|_| "http://localhost:8080".to_string())
                    }),
                    gateway_url: gateway_url.unwrap_or_else(|| {
                        std::env::var("ASH_GATEWAY_URL").unwrap_or_else(|_| "http://localhost:8081".to_string())
                    }),
                    ..Default::default()
                };
                tools::session::configure_k8s(config.clone()).await;
                ToolResult::ok(format!("Updated K8s config:\ncontrol_plane_url: {}\ngateway_url: {}", 
                    config.control_plane_url, config.gateway_url))
            }
        }
        
        // ==================== Events ====================

        Commands::Events { op } => {
            let (tool_name, args): (&str, Value) = match op {
                EventsOp::Subscribe { events, unsubscribe } => {
                    ("events_subscribe", serde_json::json!({"events": events, "unsubscribe": unsubscribe}))
                }
                EventsOp::Poll { limit, peek } => {
                    ("events_poll", serde_json::json!({"limit": limit, "peek": peek}))
                }
                EventsOp::Push { kind, source, data } => {
                    let data_val: serde_json::Value = serde_json::from_str(&data).unwrap_or(serde_json::json!({}));
                    ("events_push", serde_json::json!({"kind": kind, "source": source, "data": data_val}))
                }
            };
            // Route through gateway (events live in ash-mcp process)
            daemon::ensure_gateway().await;
            if let Some(result) = daemon::gateway_tool_call(tool_name, args.clone(), &None).await {
                result
            } else {
                tools::find_tool(tool_name).unwrap().execute(args).await
            }
        }

        // ==================== Custom Tools ====================
        
        Commands::CustomTool { op } => {
            match op {
                CustomToolOp::Create { name, script, lang } => {
                    tools::ToolRegisterTool.execute(serde_json::json!({
                        "name": name, "script": script, "lang": lang
                    })).await
                }
                CustomToolOp::List => {
                    tools::ToolListCustomTool.execute(serde_json::json!({})).await
                }
                CustomToolOp::View { name } => {
                    tools::ToolViewCustomTool.execute(serde_json::json!({"name": name})).await
                }
                CustomToolOp::Run { name, args } => {
                    tools::ToolCallCustomTool.execute(serde_json::json!({
                        "name": name, "args": args
                    })).await
                }
                CustomToolOp::Remove { name } => {
                    tools::ToolRemoveCustomTool.execute(serde_json::json!({"name": name})).await
                }
            }
        }
        
        Commands::Mcp => {
            // Run MCP server over stdio
            let all_tools = tools::all_tools();
            let server = ash::mcp::McpServer::new(all_tools);
            server.run().await.map_err(|e| anyhow::anyhow!("MCP server error: {}", e))?;
            return Ok(());
        }
        
        Commands::Gateway { op } => {
            match op {
                GatewayOp::Start { foreground } => {
                    if daemon::is_gateway_running() {
                        println!("{} Gateway is already running",
                            style::color(style::check(), style::GREEN));
                        return Ok(());
                    }

                    if foreground {
                        ash::gateway::run_gateway().await?;
                    } else {
                        // Spawn detached child with --foreground
                        let exe = std::env::current_exe()?;
                        let child = std::process::Command::new(exe)
                            .args(["gateway", "start", "--foreground"])
                            .stdin(std::process::Stdio::null())
                            .stdout(std::process::Stdio::null())
                            .stderr(std::process::Stdio::null())
                            .spawn()?;
                        println!("{} Gateway started {}",
                            style::color(style::check(), style::GREEN),
                            style::dim(&format!("(pid {})", child.id())));
                    }
                }
                GatewayOp::Stop => {
                    let pid_file = daemon::pid_path();
                    match std::fs::read_to_string(&pid_file) {
                        Ok(contents) => {
                            if let Ok(pid) = contents.trim().parse::<u32>() {
                                // Send SIGTERM via kill command
                                let _ = std::process::Command::new("kill")
                                    .arg(pid.to_string())
                                    .status();
                                // Wait briefly for cleanup
                                tokio::time::sleep(std::time::Duration::from_millis(300)).await;
                                let _ = std::fs::remove_file(daemon::pid_path());
                                let _ = std::fs::remove_file(daemon::socket_path());
                                println!("{} Gateway stopped {}",
                                    style::color(style::cross(), style::GRAY),
                                    style::dim(&format!("(pid {})", pid)));
                            } else {
                                eprintln!("Invalid PID file");
                            }
                        }
                        Err(_) => {
                            eprintln!("{} Gateway is not running",
                                style::ecolor(style::cross(), style::GRAY));
                        }
                    }
                }
                GatewayOp::Status => {
                    match daemon::gateway_call("ping", serde_json::json!({})).await {
                        Some(resp) => {
                            let uptime = resp.get("result")
                                .and_then(|r| r.get("uptime_secs"))
                                .and_then(|u| u.as_u64())
                                .unwrap_or(0);
                            println!("{} Gateway {} {}",
                                style::color(style::check(), style::GREEN),
                                style::color("running", style::GREEN),
                                style::dim(&format!("(uptime: {})", style::format_uptime(uptime))));
                        }
                        None => {
                            println!("{} Gateway {}",
                                style::color(style::cross(), style::GRAY),
                                style::dim("not running"));
                        }
                    }
                }
            }
            return Ok(());
        }

        Commands::Info => {
            let mut out = String::new();
            let ver = env!("CARGO_PKG_VERSION");

            // Banner
            out.push_str(&format!("\n{}\n", style::banner_line(ver)));

            // Try gateway for info
            if let Some(resp) = daemon::gateway_call("gateway/info", serde_json::json!({})).await {
                if let Some(info) = resp.get("result") {
                    let uptime = info.get("uptime_secs").and_then(|u| u.as_u64()).unwrap_or(0);
                    out.push_str(&format!("\n{}\n", style::section("Gateway")));
                    out.push_str(&format!("  {} {} {}\n",
                        style::color(style::check(), style::GREEN),
                        style::color("running", style::GREEN),
                        style::dim(&format!("uptime {}", style::format_uptime(uptime)))));

                    let local_mcp_port = info.get("local_mcp_port").and_then(|p| p.as_u64());
                    if let Some(port) = local_mcp_port {
                        out.push_str(&format!("  {} ash-mcp on port {}\n",
                            style::color(style::check(), style::GREEN),
                            style::color(&port.to_string(), style::CYAN)));
                    }

                    let docker_ok = info.pointer("/backends/docker").and_then(|v| v.as_bool()).unwrap_or(false);
                    let k8s_ok = info.pointer("/backends/k8s").and_then(|v| v.as_bool()).unwrap_or(false);
                    let default_backend = info.get("default_backend").and_then(|v| v.as_str()).unwrap_or("local");

                    out.push_str(&format!("\n{}\n", style::section("Backends")));
                    let docker_default = if default_backend == "docker" { " (default)" } else { "" };
                    let k8s_default = if default_backend == "k8s" { " (default)" } else { "" };
                    out.push_str(&format!("{}\n", style::status_line(
                        &format!("docker{}", docker_default), if docker_ok { "available" } else { "unavailable" }, docker_ok)));
                    out.push_str(&format!("{}\n", style::status_line(
                        &format!("k8s{}", k8s_default), if k8s_ok { "available" } else { "unavailable" }, k8s_ok)));

                    let sessions = info.get("sessions").and_then(|s| s.as_u64()).unwrap_or(0);
                    let routes = info.get("routes").and_then(|r| r.as_u64()).unwrap_or(0);
                    let tool_count = tools::all_tools().len();

                    out.push_str(&format!("\n{}\n", style::section("Stats")));
                    out.push_str(&format!("{}\n", style::kv("sessions", &format!("{} {}", sessions, style::dim(&format!("({} routes)", routes))))));
                    out.push_str(&format!("{}\n", style::kv("tools   ", &tool_count.to_string())));

                    out.push('\n');
                    print!("{}", out);
                    return Ok(());
                }
            }

            // Fallback: no gateway running
            out.push_str(&format!("\n{}\n", style::section("Gateway")));
            out.push_str(&format!("  {} {}\n",
                style::color(style::cross(), style::GRAY),
                style::dim("not running")));

            use ash::backend::BackendType;

            out.push_str(&format!("\n{}\n", style::section("Backends")));
            let manager = tools::session::BACKEND_MANAGER.read().await;
            let default_backend = manager.default_backend();
            let local_ok = manager.health_check(BackendType::Local).await.is_ok();
            let docker_ok = manager.health_check(BackendType::Docker).await.is_ok();
            let k8s_ok = manager.health_check(BackendType::K8s).await.is_ok();
            drop(manager);

            let local_default = if default_backend == BackendType::Local { " (default)" } else { "" };
            let docker_default = if default_backend == BackendType::Docker { " (default)" } else { "" };
            let k8s_default = if default_backend == BackendType::K8s { " (default)" } else { "" };
            out.push_str(&format!("{}\n", style::status_line(
                &format!("local{}", local_default), if local_ok { "available" } else { "unavailable" }, local_ok)));
            out.push_str(&format!("{}\n", style::status_line(
                &format!("docker{}", docker_default), if docker_ok { "available" } else { "unavailable" }, docker_ok)));
            out.push_str(&format!("{}\n", style::status_line(
                &format!("k8s{}", k8s_default), if k8s_ok { "available" } else { "unavailable" }, k8s_ok)));

            let tool_count = tools::all_tools().len();
            out.push_str(&format!("\n{}\n", style::section("Stats")));
            out.push_str(&format!("{}\n", style::kv("tools", &tool_count.to_string())));

            out.push('\n');
            print!("{}", out);
            return Ok(());
        }

        Commands::Tools => {
            let all = tools::all_tools();
            println!("\n{}", style::section(&format!("Tools ({})", all.len())));
            for tool in all {
                println!("{}", style::tool_entry(tool.name(), tool.description()));
            }
            println!();
            return Ok(());
        }

        Commands::Man { output_dir } => {
            use clap::CommandFactory;
            use clap_mangen::Man;
            use std::fs;
            use std::path::Path;

            let out = Path::new(&output_dir);
            fs::create_dir_all(out)?;

            let cmd = Cli::command();
            let mut count = 0;

            // Top-level: ash(1)
            let man = Man::new(cmd.clone());
            let mut buf = Vec::new();
            man.render(&mut buf)?;
            fs::write(out.join("ash.1"), buf)?;
            count += 1;

            // Subcommands: ash-grep(1), ash-edit(1), etc.
            for sub in cmd.get_subcommands() {
                if sub.get_name() == "help" { continue; }
                let name = format!("ash-{}", sub.get_name());
                let name_static: &'static str = name.clone().leak();
                let sub_cmd = sub.clone().name(name_static);
                let man = Man::new(sub_cmd.clone());
                let mut buf = Vec::new();
                man.render(&mut buf)?;
                fs::write(out.join(format!("{name}.1")), buf)?;
                count += 1;

                // Nested: ash-edit-view(1), ash-terminal-start(1), etc.
                for nested in sub_cmd.get_subcommands() {
                    if nested.get_name() == "help" { continue; }
                    let nested_name = format!("{name}-{}", nested.get_name());
                    let nested_static: &'static str = nested_name.clone().leak();
                    let nested_cmd = nested.clone().name(nested_static);
                    let man = Man::new(nested_cmd);
                    let mut buf = Vec::new();
                    man.render(&mut buf)?;
                    fs::write(out.join(format!("{nested_name}.1")), buf)?;
                    count += 1;
                }
            }

            println!("{} Generated {} man pages in {}/",
                style::color(style::check(), style::GREEN), count, output_dir);
            return Ok(());
        }

        Commands::Completions { shell } => {
            use clap::CommandFactory;
            clap_complete::generate(shell, &mut Cli::command(), "ash", &mut std::io::stdout());
            return Ok(());
        }
    };
    
    match cli.output {
        OutputFormat::Json => println!("{}", serde_json::to_string_pretty(&result)?),
        OutputFormat::Text => {
            if result.success {
                print!("{}", result.output);
                if !result.output.ends_with('\n') { println!(); }
            } else {
                let msg = result.error.unwrap_or_default();
                eprintln!("{} {}", style::ecolor("error:", style::BRIGHT_RED), msg);
                std::process::exit(1);
            }
        }
    }
    
    Ok(())
}
