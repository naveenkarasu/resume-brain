import type { AnalysisResponse } from '../api/types';

export const DEMO_RESULT: AnalysisResponse = {
  overall_score: 74,
  score_breakdown: {
    skills_match: 82,
    experience_match: 70,
    education_match: 65,
    keywords_match: 78,
  },
  missing_keywords: [
    'Kubernetes', 'Terraform', 'GraphQL', 'CI/CD pipelines',
    'System Design', 'Microservices', 'Redis',
  ],
  matched_keywords: [
    'Python', 'React', 'TypeScript', 'Docker', 'AWS',
    'PostgreSQL', 'REST APIs', 'Git', 'Agile', 'Node.js',
  ],
  keyword_density: {
    'Python': 2.14,
    'React': 1.07,
    'TypeScript': 1.07,
    'Docker': 0.71,
    'AWS': 1.43,
    'PostgreSQL': 0.71,
    'REST APIs': 0.36,
    'Git': 0.36,
    'Agile': 0.36,
    'Node.js': 0.36,
    'Kubernetes': 0.0,
    'Terraform': 0.0,
    'GraphQL': 0.0,
    'CI/CD pipelines': 0.0,
    'System Design': 0.0,
    'Microservices': 0.0,
    'Redis': 0.0,
  },
  bullet_rewrites: [
    {
      original: 'Built backend services for the platform',
      rewritten: 'Architected and deployed Python/FastAPI backend services handling 50K+ daily requests on AWS, reducing API latency by 40%',
      reason: 'Added specific technologies, metrics, and impact to match job requirements',
    },
    {
      original: 'Worked on the frontend application',
      rewritten: 'Developed responsive React/TypeScript frontend components with comprehensive test coverage, improving user engagement by 25%',
      reason: 'Specified tech stack and quantified business impact',
    },
    {
      original: 'Helped with database management',
      rewritten: 'Optimized PostgreSQL database queries and schema design, reducing query execution time by 60% across 15+ critical endpoints',
      reason: 'Added specificity with metrics and PostgreSQL keyword from job description',
    },
  ],
  ats_optimized_resume: `ALEX CHEN
Senior Software Engineer | Python, React, TypeScript, AWS
alex.chen@email.com | github.com/alexchen | linkedin.com/in/alexchen

SUMMARY
Results-driven Senior Software Engineer with 5+ years building scalable web applications using Python, React, and AWS. Proven track record of improving system performance and leading cross-functional teams in Agile environments.

EXPERIENCE
Senior Software Engineer | TechCorp Inc. | 2022 - Present
• Architected and deployed Python/FastAPI backend services handling 50K+ daily requests on AWS, reducing API latency by 40%
• Developed responsive React/TypeScript frontend components with comprehensive test coverage, improving user engagement by 25%
• Optimized PostgreSQL database queries and schema design, reducing query execution time by 60% across 15+ critical endpoints
• Led migration from monolith to Docker-containerized microservices, improving deployment frequency by 3x
• Mentored team of 4 junior developers through code reviews and pair programming sessions

Software Engineer | StartupXYZ | 2020 - 2022
• Built RESTful APIs serving 1M+ monthly active users using Node.js and Express
• Implemented real-time data pipeline processing 10K events/second using Python and AWS Lambda
• Designed and maintained CI/CD pipelines using GitHub Actions, reducing deployment time by 70%

SKILLS
Languages: Python, TypeScript, JavaScript, SQL
Frontend: React, Next.js, Tailwind CSS, HTML/CSS
Backend: FastAPI, Node.js, Express, REST APIs
Cloud: AWS (EC2, Lambda, S3, RDS), Docker
Data: PostgreSQL, MongoDB, Redis
Tools: Git, GitHub Actions, Agile/Scrum, Jira

EDUCATION
B.S. Computer Science | State University | 2020`,
  summary: 'Strong match for a Senior Full-Stack Engineer role. The candidate demonstrates solid experience with the core tech stack (Python, React, TypeScript, AWS). Key gaps are in infrastructure tooling (Kubernetes, Terraform) and system design patterns. Adding quantified achievements and missing keywords would significantly boost ATS compatibility.',
  strengths: [
    'Strong alignment with core programming languages (Python, TypeScript, React)',
    'Demonstrated experience with cloud services (AWS) and containerization (Docker)',
    'Evidence of leadership and mentoring experience',
    'Quantified achievements show measurable impact',
  ],
  weaknesses: [
    'Missing infrastructure/DevOps keywords (Kubernetes, Terraform, CI/CD)',
    'No mention of system design or architecture experience',
    'Limited evidence of microservices and distributed systems work',
    'Education section could highlight relevant coursework',
  ],
  degraded: false,
  tfidf_score: 0.42,
  semantic_score: 0.68,
  scoring_method: 'hybrid',
  section_analysis: {
    detected_sections: ['summary', 'experience', 'skills', 'education'],
    completeness: 0.69,
  },
  experience_years: 5.5,
  bullet_scores: [
    {
      text: 'Architected and deployed Python/FastAPI backend services handling 50K+ daily requests on AWS',
      has_action_verb: true,
      has_metrics: true,
      length_ok: true,
      keyword_count: 3,
      quality_score: 100,
    },
    {
      text: 'Developed responsive React/TypeScript frontend components with comprehensive test coverage',
      has_action_verb: true,
      has_metrics: false,
      length_ok: true,
      keyword_count: 2,
      quality_score: 65,
    },
    {
      text: 'Optimized PostgreSQL database queries and schema design, reducing query execution time by 60%',
      has_action_verb: true,
      has_metrics: true,
      length_ok: true,
      keyword_count: 1,
      quality_score: 90,
    },
    {
      text: 'Led migration from monolith to Docker-containerized microservices',
      has_action_verb: true,
      has_metrics: false,
      length_ok: true,
      keyword_count: 1,
      quality_score: 55,
    },
    {
      text: 'Helped with database management',
      has_action_verb: false,
      has_metrics: false,
      length_ok: false,
      keyword_count: 0,
      quality_score: 0,
    },
  ],
  education_level: 'bachelors',
};

export const DEMO_RESUME_TEXT = `Alex Chen - Software Engineer
5 years experience in web development.

Experience:
• Built backend services for the platform
• Worked on the frontend application
• Helped with database management
• Led migration to containerized deployment
• Mentored junior developers

Skills: Python, JavaScript, React, Docker, AWS, PostgreSQL, Git`;

export const DEMO_JOB_DESCRIPTION = `Senior Full-Stack Engineer

We're looking for a Senior Full-Stack Engineer to join our platform team.

Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python and TypeScript
- Experience with React and modern frontend frameworks
- Cloud experience (AWS preferred)
- Database design (PostgreSQL, Redis)
- Containerization (Docker, Kubernetes)
- Infrastructure as code (Terraform)
- GraphQL API design
- CI/CD pipeline management
- Strong system design skills
- Experience with microservices architecture
- Agile/Scrum methodology`;
