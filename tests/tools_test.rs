//! Tests for ash tools

#[cfg(test)]
mod tests {
    use ash::tools::*;
    use ash::Tool;
    use serde_json::json;

    // ==================== View Tool ====================
    
    #[tokio::test]
    async fn test_view_file() {
        // Create a temp file
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\nline5").unwrap();
        
        let result = ViewTool.execute(json!({
            "file_path": file_path.to_str().unwrap(),
            "offset": 1,
            "limit": 3
        })).await;
        
        assert!(result.success, "view should succeed");
        assert!(result.output.contains("line1"));
        assert!(result.output.contains("line2"));
        assert!(result.output.contains("line3"));
        assert!(!result.output.contains("line4"));
    }
    
    #[tokio::test]
    async fn test_view_with_offset() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3\nline4\nline5").unwrap();
        
        let result = ViewTool.execute(json!({
            "file_path": file_path.to_str().unwrap(),
            "offset": 3,
            "limit": 2
        })).await;
        
        assert!(result.success);
        assert!(!result.output.contains("line1"));
        assert!(!result.output.contains("line2"));
        assert!(result.output.contains("line3"));
        assert!(result.output.contains("line4"));
    }
    
    #[tokio::test]
    async fn test_view_nonexistent_file() {
        let result = ViewTool.execute(json!({
            "file_path": "/nonexistent/file.txt"
        })).await;
        
        assert!(!result.success);
        assert!(result.error.is_some());
    }

    // ==================== Grep Tool ====================
    
    #[tokio::test]
    async fn test_grep_files() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "hello world\nfoo bar").unwrap();
        std::fs::write(dir.path().join("b.txt"), "hello rust\nbaz qux").unwrap();
        
        let result = GrepTool.execute(json!({
            "pattern": "hello",
            "path": dir.path().to_str().unwrap()
        })).await;
        
        assert!(result.success);
        assert!(result.output.contains("hello world") || result.output.contains("hello rust"));
    }
    
    #[tokio::test]
    async fn test_grep_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "foo bar").unwrap();
        
        let result = GrepTool.execute(json!({
            "pattern": "nonexistent",
            "path": dir.path().to_str().unwrap()
        })).await;
        
        assert!(result.success);
        assert!(result.output.contains("No matches"));
    }

    // ==================== Edit Tool ====================
    
    #[tokio::test]
    async fn test_edit_view() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2\nline3").unwrap();
        
        let result = EditTool.execute(json!({
            "command": "view",
            "path": file_path.to_str().unwrap(),
            "view_range": [1, 2]
        })).await;
        
        assert!(result.success);
        assert!(result.output.contains("line1"));
        assert!(result.output.contains("line2"));
    }
    
    #[tokio::test]
    async fn test_edit_str_replace() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();
        
        let result = EditTool.execute(json!({
            "command": "str_replace",
            "path": file_path.to_str().unwrap(),
            "old_str": "world",
            "new_str": "rust"
        })).await;
        
        assert!(result.success, "str_replace should succeed: {:?}", result.error);
        
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "hello rust");
    }
    
    #[tokio::test]
    async fn test_edit_str_replace_not_unique() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "foo foo foo").unwrap();
        
        let result = EditTool.execute(json!({
            "command": "str_replace",
            "path": file_path.to_str().unwrap(),
            "old_str": "foo",
            "new_str": "bar"
        })).await;
        
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Multiple"));
    }
    
    #[tokio::test]
    async fn test_edit_str_replace_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "hello world").unwrap();
        
        let result = EditTool.execute(json!({
            "command": "str_replace",
            "path": file_path.to_str().unwrap(),
            "old_str": "nonexistent",
            "new_str": "bar"
        })).await;
        
        assert!(!result.success);
        assert!(result.error.unwrap().contains("No match"));
    }
    
    #[tokio::test]
    async fn test_edit_insert() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("test.txt");
        std::fs::write(&file_path, "line1\nline2").unwrap();
        
        let result = EditTool.execute(json!({
            "command": "insert",
            "path": file_path.to_str().unwrap(),
            "insert_line": 1,
            "insert_text": "inserted"
        })).await;
        
        assert!(result.success);
        
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("inserted"));
    }
    
    #[tokio::test]
    async fn test_edit_create() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("new_file.txt");
        
        let result = EditTool.execute(json!({
            "command": "create",
            "path": file_path.to_str().unwrap(),
            "file_text": "new content"
        })).await;
        
        assert!(result.success);
        assert!(file_path.exists());
        
        let content = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(content, "new content");
    }
    
    #[tokio::test]
    async fn test_edit_create_exists() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("existing.txt");
        std::fs::write(&file_path, "existing").unwrap();
        
        let result = EditTool.execute(json!({
            "command": "create",
            "path": file_path.to_str().unwrap(),
            "file_text": "new content"
        })).await;
        
        assert!(!result.success);
        assert!(result.error.unwrap().contains("exists"));
    }

    // ==================== Shell Tool ====================
    
    #[tokio::test]
    async fn test_shell_echo() {
        let result = ShellTool.execute(json!({
            "command": "echo hello"
        })).await;
        
        assert!(result.success);
        assert!(result.output.contains("hello"));
    }
    
    #[tokio::test]
    async fn test_shell_exit_code() {
        let result = ShellTool.execute(json!({
            "command": "exit 1"
        })).await;
        
        assert!(!result.success);
        assert!(result.error.is_some());
    }
    
    #[tokio::test]
    async fn test_shell_timeout() {
        let result = ShellTool.execute(json!({
            "command": "sleep 10",
            "timeout_secs": 1
        })).await;
        
        assert!(!result.success);
        assert!(result.error.unwrap().contains("Timeout"));
    }

    // ==================== Clipboard Tools ====================
    
    #[tokio::test]
    async fn test_clip_and_paste() {
        // Clear first
        ClearClipsTool.execute(json!({})).await;
        
        // Clip
        let result = ClipTool.execute(json!({
            "content": "test content",
            "name": "test_clip"
        })).await;
        assert!(result.success);
        
        // Paste
        let result = PasteTool.execute(json!({
            "name": "test_clip"
        })).await;
        assert!(result.success);
        assert!(result.output.contains("test content"));
        
        // List
        let result = ClipsTool.execute(json!({})).await;
        assert!(result.success);
        assert!(result.output.contains("test_clip"));
        
        // Clear specific
        let result = ClearClipsTool.execute(json!({
            "name": "test_clip"
        })).await;
        assert!(result.success);
    }
    
    #[tokio::test]
    async fn test_paste_nonexistent() {
        let result = PasteTool.execute(json!({
            "name": "nonexistent_clip_12345"
        })).await;
        
        assert!(!result.success);
    }

    // ==================== Git Tools ====================
    
    #[tokio::test]
    async fn test_git_status() {
        // This test requires being in a git repo
        let result = GitStatusTool.execute(json!({
            "short": true
        })).await;
        
        // May succeed or fail depending on environment
        // Just check it doesn't panic
        let _ = result;
    }

    // ==================== Session Tools ====================
    
    #[tokio::test]
    async fn test_session_list_empty() {
        let result = SessionListTool.execute(json!({})).await;
        assert!(result.success);
    }
    
    // Note: session_create and session_destroy require actual control plane
    // These would be integration tests
}
