import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config.settings import get_settings

settings = get_settings()


@dataclass
class LearnedWeights:
    skill_weight: float = 0.4
    vector_weight: float = 0.2
    role_weight: float = 0.2
    experience_weight: float = 0.1
    domain_weight: float = 0.1

    def normalized(self) -> dict[str, float]:
        raw = {
            "skill": max(0.0, float(self.skill_weight)),
            "vector": max(0.0, float(self.vector_weight)),
            "role": max(0.0, float(self.role_weight)),
            "experience": max(0.0, float(self.experience_weight)),
            "domain": max(0.0, float(self.domain_weight)),
        }
        total = sum(raw.values())
        if total <= 0:
            return {
                "skill": 0.4,
                "vector": 0.2,
                "role": 0.2,
                "experience": 0.1,
                "domain": 0.1,
            }
        return {key: value / total for key, value in raw.items()}


@dataclass
class LearnedJobRule:
    job_name: str
    gt_names: list[str] = field(default_factory=list)
    core_skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    domain: list[str] = field(default_factory=list)
    seniority: str = ""
    experience: list[str] = field(default_factory=list)
    hidden_rules: list[str] = field(default_factory=list)
    weights: LearnedWeights = field(default_factory=LearnedWeights)

    def summary_for_prompt(self) -> str:
        return (
            f"gt_names={self.gt_names}; core_skills={self.core_skills}; tools={self.tools}; domain={self.domain}; "
            f"seniority={self.seniority}; experience={self.experience}; hidden_rules={self.hidden_rules}; "
            f"weights={self.weights.normalized()}"
        )


def _to_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


def _normalize_key(text: str) -> str:
    return "".join(str(text).strip().lower().split())


def _parse_rule(item: dict[str, Any]) -> LearnedJobRule | None:
    job_name = str(item.get("job_name") or item.get("job") or "").strip()
    if not job_name:
        return None

    analysis = item.get("analysis", {}) if isinstance(item.get("analysis"), dict) else {}
    weights_raw = item.get("weights", {}) if isinstance(item.get("weights"), dict) else {}

    core_skills = _to_string_list(item.get("core_skills") or analysis.get("core_skills"))
    gt_names = _to_string_list(item.get("gt_names"))
    tools = _to_string_list(item.get("tools") or analysis.get("tools"))
    domain = _to_string_list(item.get("domain") or analysis.get("domain"))
    seniority = str(item.get("seniority") or analysis.get("seniority") or "").strip()
    experience = _to_string_list(item.get("experience") or analysis.get("experience"))
    hidden_rules = _to_string_list(item.get("hidden_rules") or analysis.get("hidden_rules"))

    weights = LearnedWeights(
        skill_weight=float(weights_raw.get("skill_weight", 0.4)),
        vector_weight=float(weights_raw.get("vector_weight", 0.2)),
        role_weight=float(weights_raw.get("role_weight", 0.2)),
        experience_weight=float(weights_raw.get("experience_weight", 0.1)),
        domain_weight=float(weights_raw.get("domain_weight", 0.1)),
    )

    return LearnedJobRule(
        job_name=job_name,
        gt_names=gt_names,
        core_skills=core_skills,
        tools=tools,
        domain=domain,
        seniority=seniority,
        experience=experience,
        hidden_rules=hidden_rules,
        weights=weights,
    )


class GTRuleStore:
    def __init__(self) -> None:
        self._mtime: float | None = None
        self._rules_by_name: dict[str, LearnedJobRule] = {}

    def _resolve_path(self) -> Path:
        path = Path(settings.gt_rule_file)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parent.parent
        return (project_root / path).resolve()

    def _load(self) -> None:
        self._rules_by_name = {}
        if not settings.gt_rule_enabled:
            return

        path = self._resolve_path()
        if not path.exists():
            self._mtime = None
            return

        mtime = path.stat().st_mtime
        if self._mtime == mtime and self._rules_by_name:
            return

        data = json.loads(path.read_text(encoding="utf-8"))

        items: list[dict[str, Any]] = []
        if isinstance(data, list):
            items = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            if isinstance(data.get("jobs"), list):
                items = [item for item in data["jobs"] if isinstance(item, dict)]
            elif isinstance(data.get("rules"), list):
                items = [item for item in data["rules"] if isinstance(item, dict)]
            else:
                for key, value in data.items():
                    if not isinstance(value, dict):
                        continue
                    merged = dict(value)
                    merged.setdefault("job_name", key)
                    items.append(merged)

        parsed_rules: dict[str, LearnedJobRule] = {}
        for item in items:
            rule = _parse_rule(item)
            if not rule:
                continue
            parsed_rules[_normalize_key(rule.job_name)] = rule

        self._rules_by_name = parsed_rules
        self._mtime = mtime

    def get_rule_for_job(self, job_name: str | None) -> LearnedJobRule | None:
        if not job_name:
            return None

        self._load()
        normalized = _normalize_key(job_name)
        if not normalized:
            return None

        direct = self._rules_by_name.get(normalized)
        if direct:
            return direct

        for key, rule in self._rules_by_name.items():
            if normalized in key or key in normalized:
                return rule

        return None


gt_rule_store = GTRuleStore()
