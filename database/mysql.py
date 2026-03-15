from datetime import datetime
from typing import Generator, List

from sqlalchemy import DateTime, Float, ForeignKey, Index, Integer, String, Text, create_engine, func, or_, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from config.settings import get_settings


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    industry: Mapped[str | None] = mapped_column(String(128), nullable=True)


class Candidate(Base):
    __tablename__ = "candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    location: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    years_experience: Mapped[float | None] = mapped_column(Float, nullable=True)
    salary_expectation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    degree: Mapped[str | None] = mapped_column(String(64), nullable=True)
    job_status: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    resume_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    project_experience: Mapped[str | None] = mapped_column(Text, nullable=True)
    achievements: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), index=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), nullable=True)
    location: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    salary_range: Mapped[str | None] = mapped_column(String(64), nullable=True)
    salary_min: Mapped[int | None] = mapped_column(Integer, nullable=True)
    salary_max: Mapped[int | None] = mapped_column(Integer, nullable=True)
    degree_required: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str | None] = mapped_column(String(64), index=True, nullable=True)
    job_description: Mapped[str | None] = mapped_column(Text, nullable=True)
    responsibilities: Mapped[str | None] = mapped_column(Text, nullable=True)
    preferred_qualifications: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    company: Mapped[Company | None] = relationship()


class Application(Base):
    __tablename__ = "applications"

    candidate_id: Mapped[int] = mapped_column(ForeignKey("candidates.id"), primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    application_status: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


Index("idx_candidate_location_status", Candidate.location, Candidate.job_status)
Index("idx_job_location_status", Job.location, Job.status)

settings = get_settings()
engine = create_engine(settings.mysql_url, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)

_DEGREE_ORDER = {
    "high_school": 1,
    "associate": 2,
    "bachelor": 3,
    "master": 4,
    "phd": 5,
}


def normalize_degree(degree: str | None) -> str:
    if not degree:
        return ""
    text = degree.strip().lower().replace(" ", "_")
    aliases = {
        "bs": "bachelor",
        "ba": "bachelor",
        "ms": "master",
        "ma": "master",
        "doctor": "phd",
        "ph.d": "phd",
    }
    return aliases.get(text, text)


def degree_meets_requirement(candidate_degree: str | None, required_degree: str | None) -> bool:
    if not required_degree:
        return True
    cand = _DEGREE_ORDER.get(normalize_degree(candidate_degree), 0)
    req = _DEGREE_ORDER.get(normalize_degree(required_degree), 0)
    return cand >= req


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_mysql() -> None:
    Base.metadata.create_all(bind=engine)


def get_or_create_company(db: Session, name: str, industry: str | None = None) -> Company:
    company = db.scalar(select(Company).where(Company.name == name))
    if company:
        return company
    company = Company(name=name, industry=industry)
    db.add(company)
    db.flush()
    return company


def prefilter_candidate_ids_for_job(db: Session, job: Job, limit: int) -> List[int]:
    query = select(Candidate.id, Candidate.degree).where(
        or_(Candidate.job_status.is_(None), Candidate.job_status.in_(["open", "open_to_work", "actively_looking"]))
    )

    if job.location:
        query = query.where(or_(Candidate.location == job.location, Candidate.location.is_(None)))

    if job.salary_max is not None:
        query = query.where(
            or_(Candidate.salary_expectation.is_(None), Candidate.salary_expectation <= job.salary_max)
        )

    rows = db.execute(query.limit(limit)).all()
    return [
        int(candidate_id)
        for candidate_id, degree in rows
        if degree_meets_requirement(degree, job.degree_required)
    ]


def prefilter_job_ids_for_candidate(db: Session, candidate: Candidate, limit: int) -> List[int]:
    query = select(Job.id, Job.degree_required, Job.salary_min).where(
        or_(Job.status.is_(None), Job.status.in_(["open", "published", "active"]))
    )

    if candidate.location:
        query = query.where(or_(Job.location == candidate.location, Job.location.is_(None)))

    rows = db.execute(query.limit(limit)).all()

    result: list[int] = []
    for job_id, degree_required, salary_min in rows:
        if not degree_meets_requirement(candidate.degree, degree_required):
            continue
        if candidate.salary_expectation is not None and salary_min is not None and candidate.salary_expectation > salary_min * 1.5:
            continue
        result.append(int(job_id))
    return result
