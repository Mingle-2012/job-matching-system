from database.neo4j import graph_client


DEFAULT_HIERARCHY = [
    ("react", "javascript"),
    ("nextjs", "react"),
    ("pytorch", "python"),
    ("fastapi", "python"),
    ("spring", "java"),
    ("kubernetes", "docker"),
    ("creo", "cad"),
    ("ug", "cad"),
    ("模具", "结构设计"),
    ("结构设计", "机械设计"),
]


def main() -> None:
    graph_client.init_constraints()
    for child, parent in DEFAULT_HIERARCHY:
        graph_client.add_skill_hierarchy(child, parent)
    print(f"Loaded {len(DEFAULT_HIERARCHY)} skill hierarchy edges")


if __name__ == "__main__":
    main()
