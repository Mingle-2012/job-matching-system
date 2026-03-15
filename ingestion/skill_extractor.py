import json
import re
from dataclasses import dataclass
from typing import List

from config.settings import get_settings
from services.llm_client import create_openai_client

settings = get_settings()

_SYSTEM_PROMPT = """
You are an expert recruiter for Chinese technical hiring.
Extract structured hiring profile from the text.
Return valid JSON only in this shape:
{
    "skills": ["Python", "FastAPI", "结构设计"],
    "tools": ["Creo", "UG", "CAD", "gdb"],
    "domain": ["机械设计", "MBB", "射频通信"],
    "role": ["结构工程师", "射频工程师"]
}
Rules:
- Keep skill names concise and standardized.
- Include both English and Chinese terms when appropriate.
- Output valid JSON only.
""".strip()

_DEFAULT_SKILL_LEXICON = {
    "python",
    "java",
    "golang",
    "docker",
    "kubernetes",
    "fastapi",
    "django",
    "flask",
    "react",
    "javascript",
    "typescript",
    "sql",
    "mysql",
    "postgresql",
    "redis",
    "neo4j",
    "qdrant",
    "milvus",
    "weaviate",
    "pytorch",
    "tensorflow",
    "aws",
    "azure",
    "gcp",
    "microservices",
}

_DOMAIN_SKILL_KEYWORDS = {
    "结构设计": ["结构设计", "结构开发", "结构工程", "xpm", "主设", "se", "整机结构", "手机结构"],
    "模具设计": ["模具", "注塑", "压铸", "cnc", "dfm", "公差"],
    "工业设计协同": ["id", "堆叠", "装配", "外壳", "机壳"],
    "MBB": ["mbb", "cpe", "网关", "通讯协议"],
    "驱动开发": ["驱动", "网卡驱动", "phy", "gmac", "sdk移植", "linux驱动"],
    "网络协议": ["tcpdump", "网络中断", "吞吐量", "时延", "性能调优", "pci", "pcie"],
    "射频": ["射频", "rf", "天线", "4g", "5g", "gms", "lte", "3gpp"],
    "射频调试": ["频谱仪", "示波器", "网络分析仪", "综测仪", "调试"],
    "平台调试": ["高通", "qualcomm", "mtk", "展锐"],
}

_TOOL_KEYWORDS = {
    "Creo": ["creo", "proe", "pro/e"],
    "UG": ["ug", "nx"],
    "CAD": ["cad", "autocad"],
    "SolidWorks": ["solidworks"],
    "gdb": ["gdb"],
    "tcpdump": ["tcpdump"],
    "示波器": ["示波器"],
    "频谱仪": ["频谱仪"],
    "网络分析仪": ["网络分析仪"],
    "综测仪": ["综测仪"],
}

_DOMAIN_KEYWORDS = {
    "机械设计": ["机械设计", "结构设计", "结构开发"],
    "结构设计": ["结构设计", "结构工程", "xpm", "主设", "se"],
    "模具设计": ["模具", "注塑", "压铸"],
    "MBB": ["mbb", "cpe", "网关", "通讯协议"],
    "射频通信": ["射频", "rf", "4g", "5g", "lte", "3gpp"],
    "嵌入式开发": ["驱动", "linux", "sdk", "pci", "pcie"],
}

_ROLE_KEYWORDS = {
    "结构工程师": ["结构工程师", "结构设计工程师", "结构开发", "xpm", "主设", "se"],
    "射频工程师": ["射频工程师", "射频开发", "rf工程师", "天线开发"],
    "MBB工程师": ["mbb", "网关开发", "cpe", "驱动开发"],
    "产品经理": ["产品经理", "product manager", "pm"],
    "后端工程师": ["后端", "backend", "fastapi", "django", "flask", "java后端"],
}


@dataclass
class ExtractedProfile:
    skills: list[str]
    tools: list[str]
    domain: list[str]
    role: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "skills": self.skills,
            "tools": self.tools,
            "domain": self.domain,
            "role": self.role,
        }

    def merged_terms(self) -> list[str]:
        return list(dict.fromkeys(self.skills + self.tools + self.domain + self.role))


class SkillExtractor:
    def __init__(self) -> None:
        self.client = create_openai_client()
        self.llm_enabled = self.client is not None

    def extract_profile(self, text: str, allow_llm: bool = True) -> ExtractedProfile:
        if not text:
            return ExtractedProfile(skills=[], tools=[], domain=[], role=[])

        heuristic_profile = self._extract_with_heuristic_profile(text)

        if allow_llm and self.llm_enabled and self.client:
            llm_profile = self._extract_with_llm_profile(text)
            if llm_profile:
                return self._merge_profiles(llm_profile, heuristic_profile)

        return heuristic_profile

    def extract_skills(self, text: str) -> List[str]:
        profile = self.extract_profile(text, allow_llm=True)
        return profile.merged_terms()

    def _extract_with_llm_profile(self, text: str) -> ExtractedProfile | None:
        try:
            response = self.client.chat.completions.create(
                model=settings.openai_model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text[:12000]},
                ],
            )
            payload = response.choices[0].message.content or "{}"
            data = json.loads(payload)
            return ExtractedProfile(
                skills=self._normalize(data.get("skills", [])),
                tools=self._normalize(data.get("tools", [])),
                domain=self._normalize(data.get("domain", [])),
                role=self._normalize(data.get("role", [])),
            )
        except Exception:
            self.llm_enabled = False
            return None

    def _extract_with_heuristic_profile(self, text: str) -> ExtractedProfile:
        text_lower = text.lower()
        tokens = set(re.findall(r"[A-Za-z0-9\+\.#-]+", text_lower))

        skills = [skill for skill in _DEFAULT_SKILL_LEXICON if skill in tokens]

        for canonical, keywords in _DOMAIN_SKILL_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                skills.append(canonical)

        tools: list[str] = []
        for canonical, keywords in _TOOL_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                tools.append(canonical)

        domain: list[str] = []
        for canonical, keywords in _DOMAIN_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                domain.append(canonical)

        role: list[str] = []
        for canonical, keywords in _ROLE_KEYWORDS.items():
            if any(keyword.lower() in text_lower for keyword in keywords):
                role.append(canonical)

        return ExtractedProfile(
            skills=self._normalize(skills),
            tools=self._normalize(tools),
            domain=self._normalize(domain),
            role=self._normalize(role),
        )

    def _merge_profiles(self, llm_profile: ExtractedProfile, heuristic_profile: ExtractedProfile) -> ExtractedProfile:
        return ExtractedProfile(
            skills=self._normalize(llm_profile.skills + heuristic_profile.skills),
            tools=self._normalize(llm_profile.tools + heuristic_profile.tools),
            domain=self._normalize(llm_profile.domain + heuristic_profile.domain),
            role=self._normalize(llm_profile.role + heuristic_profile.role),
        )

    @staticmethod
    def _normalize(skills: List[str]) -> List[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for skill in skills:
            cleaned = str(skill).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
        return normalized


skill_extractor = SkillExtractor()
