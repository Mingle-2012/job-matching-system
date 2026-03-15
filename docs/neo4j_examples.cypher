// Constraints
CREATE CONSTRAINT candidate_id_unique IF NOT EXISTS FOR (c:Candidate) REQUIRE c.id IS UNIQUE;
CREATE CONSTRAINT job_id_unique IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE;
CREATE CONSTRAINT skill_name_unique IF NOT EXISTS FOR (s:Skill) REQUIRE s.name IS UNIQUE;
CREATE CONSTRAINT company_id_unique IF NOT EXISTS FOR (c:Company) REQUIRE c.id IS UNIQUE;

// Skill hierarchy examples
MERGE (react:Skill {name: 'react'})
MERGE (js:Skill {name: 'javascript'})
MERGE (react)-[:SUB_SKILL_OF]->(js);

MERGE (pt:Skill {name: 'pytorch'})
MERGE (py:Skill {name: 'python'})
MERGE (pt)-[:SUB_SKILL_OF]->(py);

// Candidate -> Skill
MERGE (c:Candidate {id: 1001})
MERGE (s1:Skill {name: 'python'})
MERGE (s2:Skill {name: 'fastapi'})
MERGE (c)-[:HAS_SKILL]->(s1)
MERGE (c)-[:HAS_SKILL]->(s2);

// Job -> Skill
MERGE (j:Job {id: 2001})
MERGE (s1:Skill {name: 'python'})
MERGE (s2:Skill {name: 'docker'})
MERGE (j)-[:REQUIRES_SKILL]->(s1)
MERGE (j)-[:REQUIRES_SKILL]->(s2);

// Company relationships
MERGE (co:Company {id: 3001, name: 'Acme'})
MERGE (c:Candidate {id: 1001})
MERGE (j:Job {id: 2001})
MERGE (c)-[:WORKED_AT]->(co)
MERGE (j)-[:POSTED_BY]->(co);

// Graph retrieval: job -> candidates by skill (with sub-skill expansion)
UNWIND $job_skills AS job_skill_name
MATCH (target:Skill {name: toLower(trim(job_skill_name))})
OPTIONAL MATCH (target)<-[:SUB_SKILL_OF*0..2]-(expanded:Skill)
WITH collect(DISTINCT target.name) + collect(DISTINCT expanded.name) AS expanded_skills
UNWIND expanded_skills AS skill_name
MATCH (c:Candidate)-[:HAS_SKILL]->(s:Skill)
WHERE s.name = skill_name
RETURN c.id AS candidate_id, count(DISTINCT s) AS skill_match
ORDER BY skill_match DESC
LIMIT 100;

// Graph retrieval: candidate -> jobs by skill
UNWIND $candidate_skills AS candidate_skill_name
MATCH (source:Skill {name: toLower(trim(candidate_skill_name))})
OPTIONAL MATCH (source)-[:SUB_SKILL_OF*0..2]->(expanded:Skill)
WITH collect(DISTINCT source.name) + collect(DISTINCT expanded.name) AS expanded_skills
UNWIND expanded_skills AS skill_name
MATCH (j:Job)-[:REQUIRES_SKILL]->(s:Skill)
WHERE s.name = skill_name
RETURN j.id AS job_id, count(DISTINCT s) AS skill_match
ORDER BY skill_match DESC
LIMIT 100;
