from __future__ import annotations

from palmclaw_ubuntu.skills import SkillsLoader


def test_builtin_file_skill_is_selected(settings):
    loader = SkillsLoader(
        settings.builtin_skills_root,
        settings.workspace_skills_root,
    )
    selected = loader.select("파일을 읽고 작성해줘")
    assert [skill.name for skill in selected] == ["file-workspace"]
    assert "file_write" in loader.render(selected)


def test_workspace_skill_overrides_builtin(tmp_path):
    builtin_root = tmp_path / "builtin"
    workspace_root = tmp_path / "workspace"
    builtin = builtin_root / "same" / "SKILL.md"
    workspace = workspace_root / "same" / "SKILL.md"
    builtin.parent.mkdir(parents=True)
    workspace.parent.mkdir(parents=True)
    builtin.write_text(
        "---\nname: same\ndescription: builtin\n---\nbuiltin",
        encoding="utf-8",
    )
    workspace.write_text(
        "---\nname: same\ndescription: workspace\n---\nworkspace",
        encoding="utf-8",
    )
    loader = SkillsLoader(
        builtin_root,
        workspace_root,
    )
    same = next(skill for skill in loader.list_skills() if skill.name == "same")
    assert same.source == "workspace"
    assert same.content == "workspace"


def test_workspace_skill_symlink_outside_root_is_ignored(tmp_path):
    builtin_root = tmp_path / "builtin"
    workspace_root = tmp_path / "workspace"
    outside = tmp_path / "outside" / "SKILL.md"
    outside.parent.mkdir(parents=True)
    outside.write_text(
        "---\nname: escaped\ndescription: escaped\n---\nunsafe",
        encoding="utf-8",
    )
    link = workspace_root / "escaped"
    workspace_root.mkdir()
    link.symlink_to(outside.parent, target_is_directory=True)

    loader = SkillsLoader(builtin_root, workspace_root)

    assert loader.list_skills() == []
