all: release

release:
	cargo build --release

dev:
	cargo build
