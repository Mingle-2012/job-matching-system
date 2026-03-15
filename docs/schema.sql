CREATE DATABASE IF NOT EXISTS job_matching CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE job_matching;

CREATE TABLE IF NOT EXISTS companies (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL UNIQUE,
  industry VARCHAR(128) NULL,
  INDEX idx_company_name (name)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS candidates (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(128) NOT NULL,
  location VARCHAR(128) NULL,
  years_experience FLOAT NULL,
  salary_expectation INT NULL,
  degree VARCHAR(64) NULL,
  job_status VARCHAR(64) NULL,
  resume_summary TEXT NULL,
  project_experience TEXT NULL,
  achievements TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_candidate_name (name),
  INDEX idx_candidate_location_status (location, job_status),
  INDEX idx_candidate_salary (salary_expectation)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS jobs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  title VARCHAR(255) NOT NULL,
  company_id INT NULL,
  location VARCHAR(128) NULL,
  salary_range VARCHAR(64) NULL,
  salary_min INT NULL,
  salary_max INT NULL,
  degree_required VARCHAR(64) NULL,
  status VARCHAR(64) NULL,
  job_description TEXT NULL,
  responsibilities TEXT NULL,
  preferred_qualifications TEXT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_job_title (title),
  INDEX idx_job_location_status (location, status),
  INDEX idx_job_salary (salary_min, salary_max),
  CONSTRAINT fk_jobs_company FOREIGN KEY (company_id) REFERENCES companies(id)
) ENGINE=InnoDB;

CREATE TABLE IF NOT EXISTS applications (
  candidate_id INT NOT NULL,
  job_id INT NOT NULL,
  application_status VARCHAR(64) NOT NULL,
  created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (candidate_id, job_id),
  INDEX idx_application_status (application_status),
  CONSTRAINT fk_app_candidate FOREIGN KEY (candidate_id) REFERENCES candidates(id),
  CONSTRAINT fk_app_job FOREIGN KEY (job_id) REFERENCES jobs(id)
) ENGINE=InnoDB;
