all: release

test:
	cargo test --release

test-dev:
	cargo test

release:
	cargo build --release

dev:
	cargo build
