# CLEAN1: compliant self-test fixture

## Goal

Synthetic run tree used as the forward control for `audit_privileges.py`:
every recorded tool call is inside the calling role's contract, so the audit
must report no violation and exit 0.

## Requirements

1. The executor works only inside the workspace and reads the brief, not this spec.
2. The verifier reads the spec and the workspace but writes only the attestation.
