# Pull Request Template

## Description
Brief description of the changes and why they're needed.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Infrastructure / DevOps
- [ ] Documentation
- [ ] Refactoring

## Related Issues
Closes #

## Testing
- [ ] All tests pass: `python -m pytest tests/ -v --timeout=60`
- [ ] New tests added for new functionality
- [ ] Manual testing completed (if applicable)

## Security
- [ ] No hardcoded credentials or secrets
- [ ] Input validation applied
- [ ] CORS origins restricted
- [ ] Non-root user configured
- [ ] `.env.example` updated if env vars changed

## Infrastructure
- [ ] Docker Compose tested (if applicable)
- [ ] K8s manifests validated: `kubectl apply --dry-run=client`
- [ ] Healthcheck endpoints verified

## Checklist
- [ ] Code follows project patterns
- [ ] No deprecated API usage
- [ ] Error handling and logging added
- [ ] Context files updated (`.opencode/context/`)
- [ ] Commit message follows convention

## Screenshots (if UI change)
N/A
