//! Send commands to a running game and print what it says back.
//!
//! Deliberately dumb: it writes a line to a socket and reads a line of JSON. Sharing no
//! code with the game is what lets this be a second binary without splitting the crate
//! into a lib — the protocol is text, so there is nothing to share.
//!
//! Blocking is the feature. The game holds each reply until the command has really
//! happened, so `wusel-ctl wait plan` returns when the plan is done and `capture` when
//! the PNG is on disk.
//!
//! `run <scenario>` is handled entirely here rather than in the game: a scenario is just
//! these same commands in a file, so replaying it client-side keeps the protocol at one
//! command per connection and keeps a script format out of the engine.

#[cfg(not(target_arch = "wasm32"))]
fn main() -> std::process::ExitCode {
    driver::main()
}

/// The client is Unix-socket shaped, and `just check-web` checks every binary target.
#[cfg(target_arch = "wasm32")]
fn main() {}

#[cfg(not(target_arch = "wasm32"))]
mod driver {
    use std::{
        io::{BufRead, BufReader, Write},
        os::unix::net::UnixStream,
        process::ExitCode,
    };

    const SOCKET_ENV: &str = "WUSEL_CONTROL";

    pub(super) fn main() -> ExitCode {
        let args: Vec<String> = std::env::args().skip(1).collect();
        if args.is_empty() {
            eprintln!("usage: wusel-ctl <command> [args...]");
            eprintln!("       wusel-ctl run <scenario file>");
            eprintln!("socket comes from ${SOCKET_ENV}");
            return ExitCode::FAILURE;
        }

        let Ok(socket) = std::env::var(SOCKET_ENV) else {
            eprintln!("${SOCKET_ENV} is not set; start the game with it to open a socket");
            return ExitCode::FAILURE;
        };

        if args[0] == "run" {
            let Some(path) = args.get(1) else {
                eprintln!("run needs a scenario file");
                return ExitCode::FAILURE;
            };
            return scenario(&socket, path);
        }

        match send(&socket, &args.join(" ")) {
            Ok(reply) => {
                print!("{reply}");
                verdict(&reply)
            }
            Err(error) => {
                eprintln!("{error}");
                ExitCode::FAILURE
            }
        }
    }

    /// Replays a scenario line by line, stopping at the first failure — a scenario is a
    /// sequence, so carrying on after a step that did not happen would report on a world
    /// that never existed.
    fn scenario(socket: &str, path: &str) -> ExitCode {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(error) => {
                eprintln!("cannot read {path}: {error}");
                return ExitCode::FAILURE;
            }
        };

        let mut replies: Vec<String> = Vec::new();
        let mut failed = false;

        for line in text.lines() {
            let command = line.split('#').next().unwrap_or("").trim();
            if command.is_empty() {
                continue;
            }
            eprintln!("> {command}");
            match send(socket, command) {
                Ok(reply) => {
                    let ok = reply.contains("\"ok\":true");
                    replies.push(entry(command, reply.trim()));
                    if !ok {
                        failed = true;
                        break;
                    }
                }
                Err(error) => {
                    replies.push(entry(
                        command,
                        &format!("{{\"ok\":false,\"error\":{error:?}}}"),
                    ));
                    failed = true;
                    break;
                }
            }
        }

        println!(
            "{{\"scenario\":{:?},\"ok\":{},\"steps\":[{}]}}",
            path,
            !failed,
            replies.join(",")
        );
        if failed {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        }
    }

    /// One command per connection: connect, say it, wait for the single line back.
    fn send(socket: &str, command: &str) -> Result<String, String> {
        let mut stream = UnixStream::connect(socket)
            .map_err(|error| format!("no game listening on {socket}: {error}"))?;

        writeln!(stream, "{command}").map_err(|error| format!("could not send: {error}"))?;

        let mut reply = String::new();
        BufReader::new(&stream)
            .read_line(&mut reply)
            .map_err(|error| format!("no reply: {error}"))?;
        Ok(reply)
    }

    fn entry(command: &str, reply: &str) -> String {
        format!("{{\"step\":{command:?},\"reply\":{reply}}}")
    }

    /// The exit status carries the verdict too, so a caller can branch without parsing.
    fn verdict(reply: &str) -> ExitCode {
        if reply.contains("\"ok\":true") {
            ExitCode::SUCCESS
        } else {
            ExitCode::FAILURE
        }
    }
}
