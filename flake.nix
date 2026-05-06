{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        pythonEnv = pkgs.python3.withPackages (
          ps: with ps; [
            # Runtime
            rapidfuzz
            voluptuous
            # Test
            pytest
            pytest-asyncio
            pyyaml
          ]
        );

        devTools = with pkgs; [
          ruff
          mypy
        ];

        fmtTools = with pkgs; [
          treefmt
          nixfmt
          taplo
          ruff
          python3Packages.mdformat
          prettier
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          name = "closest-intent-dev";
          packages = [ pythonEnv ] ++ devTools ++ fmtTools;
        };

        # `nix flake check` runs the test suite in a sandbox. CI can
        # call this directly without needing a separate test runner.
        checks = {
          tests =
            pkgs.runCommand "closest-intent-tests"
              {
                nativeBuildInputs = [ pythonEnv ];
                src = self;
              }
              ''
                cp -r $src/. ./work
                chmod -R u+w ./work
                cd ./work
                export PYTEST_CACHE_DIR="$TMPDIR/pytest-cache"
                ${pythonEnv}/bin/pytest tests/ -v -o cache_dir="$PYTEST_CACHE_DIR"
                touch $out
              '';

          lint =
            pkgs.runCommand "closest-intent-lint"
              {
                nativeBuildInputs = [ pkgs.ruff ];
                src = self;
              }
              ''
                cp -r $src/. ./work
                chmod -R u+w ./work
                cd ./work
                export RUFF_CACHE_DIR="$TMPDIR/ruff-cache"
                ${pkgs.ruff}/bin/ruff check .
                ${pkgs.ruff}/bin/ruff format --check .
                touch $out
              '';

          fmt =
            pkgs.runCommand "closest-intent-fmt"
              {
                nativeBuildInputs = fmtTools;
                src = self;
              }
              ''
                cp -r $src/. ./work
                chmod -R u+w ./work
                cd ./work
                export HOME="$TMPDIR"
                treefmt --ci --no-cache
                touch $out
              '';
        };

        # `nix fmt` runs treefmt across all configured file types.
        formatter = pkgs.writeShellApplication {
          name = "treefmt";
          runtimeInputs = fmtTools;
          text = ''exec treefmt "$@"'';
        };
      }
    );
}
