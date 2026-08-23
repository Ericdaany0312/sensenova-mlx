# Publishing to GitHub

This checklist stops local preparation from being confused with an external
publication. Creating the repository, pushing code, tagging a version, and
publishing a Release require the repository owner's explicit approval.

## 1. Decide the publication identity

Before any network-changing command, confirm:

- GitHub owner or organization.
- Repository name, recommended: `sensenova-mlx`.
- Public or private visibility.
- Whether the Chinese paper and PowerPoint should be attached to the Release.
- The author name and email that may become public in Git history.

Recommended repository metadata:

- Description: `Local MLX text-to-image runtime for SenseNova U1.5 on Apple silicon.`
- Topics: `mlx`, `apple-silicon`, `sensenova`, `text-to-image`,
  `local-inference`, `macos`, `python`.

## 2. Create an empty GitHub repository

Create the repository without GitHub-generated README, license, or `.gitignore`,
because all three already exist locally. Do not upload model weights or use Git
LFS for this project.

GitHub's official local-source import guide:
https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github

## 3. Connect and push only after approval

From the repository root, replace the placeholders only after the owner and
visibility are confirmed:

```bash
git remote add origin https://github.com/<owner>/sensenova-mlx.git
git remote -v
git push -u origin main
```

Then verify the rendered README, sample image, license detection, CI result, and
the complete GitHub file tree before creating a tag.

## 4. Enable repository protections

For a public repository, enable at minimum:

- Dependabot alerts and dependency graph.
- Secret scanning and push protection.
- Code scanning with the GitHub default setup where available.
- A `main` branch rule requiring the macOS CI job for pull requests.
- Private vulnerability reporting through GitHub Security Advisories.

GitHub security settings reference:
https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-security-and-analysis-settings-for-your-repository

## 5. Publish v0.2.0 only after main is verified

1. Re-run the tests, build, package audit, secret scan, and checksums.
2. Confirm that `git status` is clean and GitHub CI passes.
3. Create an annotated `v0.2.0` tag locally.
4. Push that tag.
5. Create a GitHub Release from [`releases/v0.2.0.md`](releases/v0.2.0.md).
6. Attach only the approved items in
   [`releases/v0.2.0-assets.md`](releases/v0.2.0-assets.md).
7. Download every uploaded asset once and verify its SHA-256.

Tagging, pushing, and publishing are intentionally not automated by this
repository. The 47.537 GiB formal weights remain official external downloads.
