//! Build script: generates man pages and shell completions at build time.
//!
//! Man pages go to $OUT_DIR/man/ and completions to $OUT_DIR/completions/.
//! Only runs in release builds or when ASH_GEN_ARTIFACTS=1 is set.
//!
//! The CLI definitions are included directly from src/cli.rs via include!()
//! so this script has no dependency on the ash library itself.

include!("src/cli.rs");

fn main() {
    // Only generate in release builds or when explicitly requested
    let gen = std::env::var("ASH_GEN_ARTIFACTS").is_ok()
        || std::env::var("PROFILE").map(|p| p == "release").unwrap_or(false);

    if !gen {
        println!("cargo:rerun-if-changed=src/cli.rs");
        return;
    }

    let out_dir = std::env::var("OUT_DIR").unwrap();
    let out_dir = std::path::Path::new(&out_dir);

    generate_man_pages(out_dir);
    generate_completions(out_dir);
}

fn generate_man_pages(out_dir: &std::path::Path) {
    use clap::CommandFactory;
    use clap_mangen::Man;

    let man_dir = out_dir.join("man");
    std::fs::create_dir_all(&man_dir).unwrap();

    let cmd = Cli::command();

    // Top-level: ash(1)
    let man = Man::new(cmd.clone());
    let mut buf = Vec::new();
    man.render(&mut buf).unwrap();
    std::fs::write(man_dir.join("ash.1"), buf).unwrap();

    // Subcommands: ash-grep(1), ash-edit(1), etc.
    for sub in cmd.get_subcommands() {
        if sub.get_name() == "help" {
            continue;
        }
        let name = format!("ash-{}", sub.get_name());
        let name_static: &'static str = name.clone().leak();
        let sub_cmd = sub.clone().name(name_static);
        let man = Man::new(sub_cmd.clone());
        let mut buf = Vec::new();
        man.render(&mut buf).unwrap();
        std::fs::write(man_dir.join(format!("{name}.1")), buf).unwrap();

        // Nested: ash-edit-view(1), ash-terminal-start(1), etc.
        for nested in sub_cmd.get_subcommands() {
            if nested.get_name() == "help" {
                continue;
            }
            let nested_name = format!("{name}-{}", nested.get_name());
            let nested_static: &'static str = nested_name.clone().leak();
            let nested_cmd = nested.clone().name(nested_static);
            let man = Man::new(nested_cmd);
            let mut buf = Vec::new();
            man.render(&mut buf).unwrap();
            std::fs::write(man_dir.join(format!("{nested_name}.1")), buf).unwrap();
        }
    }

    println!("cargo:rerun-if-changed=src/cli.rs");
}

fn generate_completions(out_dir: &std::path::Path) {
    use clap::CommandFactory;
    use clap_complete::{generate_to, Shell};

    let comp_dir = out_dir.join("completions");
    std::fs::create_dir_all(&comp_dir).unwrap();

    let mut cmd = Cli::command();
    for shell in [Shell::Bash, Shell::Zsh, Shell::Fish, Shell::Elvish, Shell::PowerShell] {
        let _ = generate_to(shell, &mut cmd, "ash", &comp_dir);
    }

    println!("cargo:rerun-if-changed=src/cli.rs");
}
