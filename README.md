# GitHub Codespace Environment for the Data Engineering Class

## User Notes

Start a GitHub Codespace from this repository to get all the resources required to do the hands-on exercises.

## Developer Notes: Dev Container Configuration

The `.devcontainer/devcontainer.json` configures the Codespace environment. Key documentation:

- [Introduction to dev containers](https://docs.github.com/en/codespaces/setting-up-your-project-for-codespaces/adding-a-dev-container-configuration/introduction-to-dev-containers)
- [Universal dev container image](https://github.com/devcontainers/images/tree/main/src/universal)
- [Available dev container features](https://containers.dev/features)

### Configuration Choices

**Base Image:** `mcr.microsoft.com/devcontainers/universal:4.1.1`
- GitHub's default Codespace image
- Includes Python, Node.js, Docker, Git, and common dev tools
- See [universal image docs](https://github.com/devcontainers/images/tree/main/src/universal) for full list

**Features:** Prefer dev container features over manual installation scripts
```json
"features": {
  "ghcr.io/dhoeric/features/google-cloud-cli:1": {}
}
```
- Features are self-contained, versioned installation units
- Browse available features at [containers.dev/features](https://containers.dev/features)
- Cleaner than manual apt installation and handles updates automatically

**Lifecycle Commands:**
| Command | When it runs | Use case |
|---------|--------------|----------|
| `onCreateCommand` | Container creation | One-time setup |
| `updateContentCommand` | After create + on rebuild | Cached operations |
| `postCreateCommand` | After container ready | Final setup steps |
| `postStartCommand` | Every container start | Runtime config |

We use `postCreateCommand` for additional apt packages not covered by features.

**VS Code Extensions:**
```json
"customizations": {
  "vscode": {
    "extensions": ["ms-toolsai.jupyter", "ms-python.python"]
  }
}
```

### Updating the Configuration

1. Edit `.devcontainer/devcontainer.json`
2. Test by rebuilding: Command Palette > "Codespaces: Rebuild Container"
3. For new features, search [containers.dev/features](https://containers.dev/features)
