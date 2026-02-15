//! Code outline tool - show code structure using tree-sitter
//!
//! Displays class/function/method hierarchy with line numbers,
//! like IDE code folding but as text output.

use crate::{BoxFuture, Tool, ToolResult};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tree_sitter::{Language, Parser, Tree, Node};

/// A symbol in the code outline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Symbol {
    pub name: String,
    pub kind: SymbolKind,
    pub line: usize,
    pub end_line: usize,
    pub children: Vec<Symbol>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Class,
    Function,
    Method,
    Struct,
    Enum,
    Interface,
    Trait,
    Impl,
    Module,
    Constant,
    Variable,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolKind::Class => write!(f, "class"),
            SymbolKind::Function => write!(f, "fn"),
            SymbolKind::Method => write!(f, "method"),
            SymbolKind::Struct => write!(f, "struct"),
            SymbolKind::Enum => write!(f, "enum"),
            SymbolKind::Interface => write!(f, "interface"),
            SymbolKind::Trait => write!(f, "trait"),
            SymbolKind::Impl => write!(f, "impl"),
            SymbolKind::Module => write!(f, "mod"),
            SymbolKind::Constant => write!(f, "const"),
            SymbolKind::Variable => write!(f, "var"),
        }
    }
}

/// Supported languages for outline extraction
#[derive(Debug, Clone, Copy)]
pub enum OutlineLanguage {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Go,
    C,
    Cpp,
    Java,
    Bash,
}

impl OutlineLanguage {
    /// Detect language from file extension
    pub fn from_extension(ext: &str) -> Option<Self> {
        match ext.to_lowercase().as_str() {
            "py" | "pyi" => Some(Self::Python),
            "rs" => Some(Self::Rust),
            "js" | "mjs" | "cjs" => Some(Self::JavaScript),
            "ts" | "tsx" | "mts" => Some(Self::TypeScript),
            "go" => Some(Self::Go),
            "c" | "h" => Some(Self::C),
            "cpp" | "cc" | "cxx" | "hpp" | "hxx" => Some(Self::Cpp),
            "java" => Some(Self::Java),
            "sh" | "bash" => Some(Self::Bash),
            _ => None,
        }
    }

    /// Get tree-sitter language
    fn tree_sitter_language(&self) -> Language {
        match self {
            Self::Python => tree_sitter_python::LANGUAGE.into(),
            Self::Rust => tree_sitter_rust::LANGUAGE.into(),
            Self::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
            Self::TypeScript => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            Self::Go => tree_sitter_go::LANGUAGE.into(),
            Self::C => tree_sitter_c::LANGUAGE.into(),
            Self::Cpp => tree_sitter_cpp::LANGUAGE.into(),
            Self::Java => tree_sitter_java::LANGUAGE.into(),
            Self::Bash => tree_sitter_bash::LANGUAGE.into(),
        }
    }
}

/// Extract code outline from a file
pub fn extract_outline(file_path: &Path) -> Result<Vec<Symbol>> {
    let ext = file_path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| anyhow!("Cannot determine file extension"))?;

    let lang = OutlineLanguage::from_extension(ext)
        .ok_or_else(|| anyhow!("Unsupported language for extension: {}", ext))?;

    let source = std::fs::read_to_string(file_path)?;
    extract_outline_from_source(&source, lang)
}

/// Extract outline from source code string
pub fn extract_outline_from_source(source: &str, lang: OutlineLanguage) -> Result<Vec<Symbol>> {
    let mut parser = Parser::new();
    parser.set_language(&lang.tree_sitter_language())?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow!("Failed to parse source code"))?;

    let symbols = extract_symbols(&tree, source.as_bytes(), lang);
    Ok(symbols)
}

fn extract_symbols(tree: &Tree, source: &[u8], lang: OutlineLanguage) -> Vec<Symbol> {
    let root = tree.root_node();
    extract_symbols_from_node(root, source, lang, 0)
}

fn extract_symbols_from_node(node: Node, source: &[u8], lang: OutlineLanguage, depth: usize) -> Vec<Symbol> {
    let mut symbols = Vec::new();

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(symbol) = node_to_symbol(child, source, lang, depth) {
            symbols.push(symbol);
        } else {
            // Recurse into non-symbol nodes (e.g., module body)
            let nested = extract_symbols_from_node(child, source, lang, depth);
            symbols.extend(nested);
        }
    }

    symbols
}

fn node_to_symbol(node: Node, source: &[u8], lang: OutlineLanguage, depth: usize) -> Option<Symbol> {
    let kind = match lang {
        OutlineLanguage::Python => python_symbol_kind(&node),
        OutlineLanguage::Rust => rust_symbol_kind(&node),
        OutlineLanguage::JavaScript | OutlineLanguage::TypeScript => js_symbol_kind(&node),
        OutlineLanguage::Go => go_symbol_kind(&node),
        OutlineLanguage::C | OutlineLanguage::Cpp => c_symbol_kind(&node),
        OutlineLanguage::Java => java_symbol_kind(&node),
        OutlineLanguage::Bash => bash_symbol_kind(&node),
    }?;

    let name = extract_name(node, source, lang)?;
    let line = node.start_position().row + 1;
    let end_line = node.end_position().row + 1;

    // Extract children (methods inside class, etc.)
    let children = extract_symbols_from_node(node, source, lang, depth + 1);

    Some(Symbol {
        name,
        kind,
        line,
        end_line,
        children,
    })
}

fn extract_name(node: Node, source: &[u8], lang: OutlineLanguage) -> Option<String> {
    // Find the name/identifier child node
    let name_node = match lang {
        OutlineLanguage::Python => {
            node.child_by_field_name("name")
        }
        OutlineLanguage::Rust => {
            node.child_by_field_name("name")
                .or_else(|| {
                    // For impl blocks, get the type name
                    if node.kind() == "impl_item" {
                        node.child_by_field_name("type")
                    } else {
                        None
                    }
                })
        }
        OutlineLanguage::JavaScript | OutlineLanguage::TypeScript => {
            node.child_by_field_name("name")
                .or_else(|| {
                    // Arrow functions assigned to variables
                    if node.kind() == "lexical_declaration" || node.kind() == "variable_declaration" {
                        node.named_child(0)?.child_by_field_name("name")
                    } else {
                        None
                    }
                })
        }
        OutlineLanguage::Go => {
            node.child_by_field_name("name")
        }
        OutlineLanguage::C | OutlineLanguage::Cpp => {
            node.child_by_field_name("declarator")
                .and_then(|d| {
                    // Handle function declarators
                    if d.kind() == "function_declarator" {
                        d.child_by_field_name("declarator")
                    } else {
                        Some(d)
                    }
                })
                .or_else(|| node.child_by_field_name("name"))
        }
        OutlineLanguage::Java => {
            node.child_by_field_name("name")
        }
        OutlineLanguage::Bash => {
            node.child_by_field_name("name")
        }
    }?;

    let name = name_node.utf8_text(source).ok()?;
    Some(name.to_string())
}

// Language-specific symbol kind detection

fn python_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "class_definition" => Some(SymbolKind::Class),
        "function_definition" => {
            // Check if it's inside a class (method) by looking at parent
            let parent = node.parent()?;
            if parent.kind() == "block" {
                let grandparent = parent.parent()?;
                if grandparent.kind() == "class_definition" {
                    return Some(SymbolKind::Method);
                }
            }
            Some(SymbolKind::Function)
        }
        _ => None,
    }
}

fn rust_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "function_item" => Some(SymbolKind::Function),
        "struct_item" => Some(SymbolKind::Struct),
        "enum_item" => Some(SymbolKind::Enum),
        "trait_item" => Some(SymbolKind::Trait),
        "impl_item" => Some(SymbolKind::Impl),
        "mod_item" => Some(SymbolKind::Module),
        "const_item" => Some(SymbolKind::Constant),
        "static_item" => Some(SymbolKind::Variable),
        _ => None,
    }
}

fn js_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "class_declaration" => Some(SymbolKind::Class),
        "function_declaration" => Some(SymbolKind::Function),
        "method_definition" => Some(SymbolKind::Method),
        "arrow_function" => Some(SymbolKind::Function),
        "interface_declaration" => Some(SymbolKind::Interface),
        _ => None,
    }
}

fn go_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "function_declaration" => Some(SymbolKind::Function),
        "method_declaration" => Some(SymbolKind::Method),
        "type_declaration" => Some(SymbolKind::Struct), // Could be struct or interface
        "const_declaration" => Some(SymbolKind::Constant),
        "var_declaration" => Some(SymbolKind::Variable),
        _ => None,
    }
}

fn c_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "function_definition" => Some(SymbolKind::Function),
        "struct_specifier" => Some(SymbolKind::Struct),
        "enum_specifier" => Some(SymbolKind::Enum),
        "class_specifier" => Some(SymbolKind::Class), // C++
        _ => None,
    }
}

fn java_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "class_declaration" => Some(SymbolKind::Class),
        "method_declaration" => Some(SymbolKind::Method),
        "interface_declaration" => Some(SymbolKind::Interface),
        "enum_declaration" => Some(SymbolKind::Enum),
        "constructor_declaration" => Some(SymbolKind::Method),
        _ => None,
    }
}

fn bash_symbol_kind(node: &Node) -> Option<SymbolKind> {
    match node.kind() {
        "function_definition" => Some(SymbolKind::Function),
        _ => None,
    }
}

/// Format outline as text with tree structure
pub fn format_outline(symbols: &[Symbol], indent: usize) -> String {
    let mut output = String::new();
    let total = symbols.len();

    for (i, symbol) in symbols.iter().enumerate() {
        let is_last = i == total - 1;
        let prefix = if indent == 0 {
            String::new()
        } else {
            let mut p = "  ".repeat(indent - 1);
            p.push_str(if is_last { "└─ " } else { "├─ " });
            p
        };

        let line_info = if symbol.line == symbol.end_line {
            format!("L{}", symbol.line)
        } else {
            format!("L{}-{}", symbol.line, symbol.end_line)
        };

        output.push_str(&format!(
            "{}{} {} ({})\n",
            prefix, symbol.kind, symbol.name, line_info
        ));

        if !symbol.children.is_empty() {
            output.push_str(&format_outline(&symbol.children, indent + 1));
        }
    }

    output
}

// ============== Tool Implementation ==============

#[derive(Debug, Deserialize)]
pub struct OutlineArgs {
    pub file_path: String,
    #[serde(default)]
    pub session_id: Option<String>,
}

pub struct OutlineTool;

impl Tool for OutlineTool {
    fn name(&self) -> &'static str { "code_outline" }
    
    fn description(&self) -> &'static str { 
        "Show code structure (classes, functions, methods) with line numbers. Like IDE folding but as text."
    }
    
    fn schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "File path to analyze"},
                "session_id": {"type": "string", "description": "Execute in session sandbox"}
            },
            "required": ["file_path"]
        })
    }
    
    fn execute(&self, args: serde_json::Value) -> BoxFuture<'_, ToolResult> {
        Box::pin(async move {
            let args: OutlineArgs = match serde_json::from_value(args) {
                Ok(a) => a,
                Err(e) => return ToolResult::err(format!("Invalid args: {e}")),
            };

            // TODO: Handle session_id for sandbox execution
            let path = Path::new(&args.file_path);
            
            match extract_outline(path) {
                Ok(symbols) => {
                    if symbols.is_empty() {
                        ToolResult::ok("No symbols found in file".to_string())
                    } else {
                        let output = format_outline(&symbols, 0);
                        ToolResult::ok(output)
                    }
                }
                Err(e) => ToolResult::err(format!("{e}")),
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_outline() {
        let source = r#"
class Foo:
    def __init__(self, x):
        self.x = x

    def process(self):
        return self.x * 2

def helper():
    pass
"#;
        let symbols = extract_outline_from_source(source, OutlineLanguage::Python).unwrap();
        assert!(!symbols.is_empty());
        println!("{}", format_outline(&symbols, 0));
    }

    #[test]
    fn test_rust_outline() {
        let source = r#"
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

fn main() {
    let p = Point::new(0.0, 0.0);
}
"#;
        let symbols = extract_outline_from_source(source, OutlineLanguage::Rust).unwrap();
        assert!(!symbols.is_empty());
        println!("{}", format_outline(&symbols, 0));
    }
}
