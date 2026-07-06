# Code Review Process

## Pre-Review Checklist
- [ ] Load context files from `.opencode/context/`
- [ ] Run full test suite: `python -m pytest tests/ -v --timeout=60`
- [ ] No deprecated API usage (ES `body=`, pymilvus ORM, etc.)
- [ ] Security hardening applied (non-root user, CORS, input validation)
- [ ] No hardcoded credentials or secrets
- [ ] `.env` changes are reflected in `.env.example`

## Review Areas

### Security
- Check for hardcoded API keys, passwords, tokens
- Verify CORS origins are restricted
- Validate input sanitization
- Confirm non-root user in Docker/K8s
- Check capability drops and privilege escalation settings

### Code Quality
- Follow modular, functional patterns consistent with project
- No deprecated library APIs
- Proper error handling and logging
- Type hints present and correct
- No commented-out code

### Infrastructure
- Docker: non-root user, multi-stage build, healthcheck
- K8s: securityContext, PDB, HPA, resource limits
- docker-compose: env vars for secrets (no hardcoding)
- CI/CD: tests run on push/PR

### Testing
- New features have corresponding tests
- Existing tests still pass
- Edge cases covered (empty input, error states, rate limits)

## API Key Handling
- Never commit real API keys
- `.env` is in `.gitignore`
- `.env.example` has placeholder values
- CI/CD uses GitHub Secrets for API keys

## Completion Criteria
- All tests pass (85+ passing, 0 failures)
- No security warnings
- Code committed with meaningful message
- Context files updated if project structure changed
