# BluePeak Technologies — IT Service Desk Runbook

**Document Type:** IT Operations and Troubleshooting Runbook  
**Customer:** BluePeak Technologies  
**Department:** IT Service Desk and Operations  
**Audience:** IT support engineers, service desk analysts, and authorized technical personnel  
**Purpose:** Standardize troubleshooting, incident triage, ticket prioritization, and escalation.

> **Note:** BluePeak Technologies is a fictional customer for this project. Procedures and priority definitions are illustrative and must be validated before production use.

---

## 1. Email Troubleshooting

### Objective

Diagnose email delivery, access, synchronization, and availability issues while determining whether the problem affects an individual user or multiple users.

### Initial Assessment

When a user cannot send or receive email:

1. Confirm the user's email address and the approximate time the issue began.
2. Determine whether the issue affects one user, a specific team, or the entire organization.
3. Check network connectivity and access to other corporate applications.
4. Review the approved email service status dashboard for outages or degraded service.
5. Record any error messages, delivery failures, or synchronization problems.

### Single-User Troubleshooting

If the issue affects only one user:

- Sign out of the email application and sign back in.
- Test access through the approved webmail client.
- Verify that the mailbox has not reached its storage quota.
- Check for incorrect account settings or authentication errors.
- Review whether messages are stuck in the Outbox or failing to synchronize.
- Restart the application if the issue appears limited to the desktop client.

Do not ask users to reinstall software before checking service health and simpler causes.

### Widespread Email Incidents

If multiple users report similar issues:

1. Check the service status dashboard and existing incident tickets.
2. Compare the reported symptoms and approximate start times.
3. Open or update a shared incident record.
4. Escalate the suspected service outage to the responsible messaging or infrastructure team.
5. Communicate verified updates through the approved incident communication channel.

Avoid asking every affected user to reinstall applications or change local settings when evidence indicates a centralized service issue.

### Escalation Criteria

Escalate when the issue affects multiple teams, prevents access to business-critical communications, persists after standard troubleshooting, or appears to involve an authentication or service availability incident.

**Expected outcome:** Determine the likely scope, document findings, and route the issue to the correct support team.

---

## 2. Laptop Performance Troubleshooting

### Objective

Identify common causes of slow laptop performance while maintaining endpoint security and protecting user data.

### Initial Diagnostics

For a slow or unresponsive laptop, inspect the following:

- **CPU utilization:** Identify processes consistently consuming excessive processor resources.
- **Memory usage:** Check whether available memory is low or applications are consuming unusually high amounts.
- **Storage capacity:** Verify available disk space and investigate storage-related warnings.
- **Startup applications:** Identify unnecessary startup programs using approved operating-system controls.
- **Operating system updates:** Check whether updates are pending, failing, or awaiting a restart.
- **Endpoint security:** Review the status of approved endpoint protection and security scans.
- **Hardware health:** Check available disk, memory, battery, and manufacturer diagnostic results.
- **Thermal and power conditions:** Look for overheating symptoms, fan problems, or power-related performance limitations.

### Standard Troubleshooting Procedure

1. Ask when the performance issue began and whether it occurs continuously or intermittently.
2. Determine whether the issue affects the entire operating system or only a particular application.
3. Review CPU, memory, and storage utilization.
4. Close unnecessary applications using normal operating-system controls.
5. Remove unnecessary startup applications where permitted by policy.
6. Check for approved operating-system updates and complete required restarts.
7. Run authorized hardware diagnostics when symptoms indicate a possible hardware issue.
8. Record the results and confirm whether performance improves.

**Security restriction:** Never disable endpoint protection, antivirus, device management, or other security agents as a shortcut to improving performance.

### Escalation Criteria

Escalate to desktop engineering or hardware support when:

- Hardware diagnostics report a failure.
- The laptop repeatedly shuts down unexpectedly after basic software remediation.
- The device shows signs of storage failure or possible data loss.
- Performance remains severely degraded after approved troubleshooting.
- The issue affects several devices and may indicate a broader software or management problem.

If hardware failure or data loss is suspected, avoid unnecessary diagnostic actions that could worsen the situation. Follow the approved device recovery process.

**Expected outcome:** Identify the likely cause, apply approved remediation, and escalate unresolved hardware or system issues.

---

## 3. IT Service Desk Ticket Priority

### Objective

Assign incident priorities based on business impact, urgency, security implications, and the availability of workarounds.

### Priority Definitions

| Priority | Definition | Example | Handling guidance |
|---|---|---|---|
| **P1 — Critical** | Company-wide outage or critical security incident | Corporate authentication unavailable across the organization; confirmed major security compromise | Initiate the designated urgent escalation and incident response process |
| **P2 — High** | A major team or business function is blocked with no reasonable workaround | A department cannot access a critical business application | Escalate to the responsible technical team and track restoration |
| **P3 — Medium** | An individual user is blocked or significantly affected, with a limited workaround | One employee cannot access a required application but has restricted alternative access | Troubleshoot and route through the standard incident process |
| **P4 — Low** | Standard request, information request, or low-impact issue | Software installation request or general IT question | Handle through the routine service request queue |

### Priority Assignment Rules

1. Determine the number of users and business functions affected.
2. Identify whether a critical service is unavailable.
3. Assess whether users can continue working through an approved workaround.
4. Determine whether the incident involves a suspected security compromise.
5. Assign the priority that best reflects the verified impact.
6. Reassess the priority if the scope or business impact changes.

### Security Incident Considerations

A suspected critical security incident must be reported through the designated Security escalation channel promptly. Do not delay reporting while completing routine troubleshooting.

Not every security-related ticket is automatically P1. The responsible incident management or Security team should assess severity according to the approved incident response policy.

### Important Notes

- Ticket priority reflects business impact and urgency, not simply the order in which tickets arrive.
- A large number of similar reports may indicate a centralized incident.
- Response targets and service-level agreements must be defined by BluePeak Technologies; this runbook does not establish specific response-time guarantees.
- Document the reason for the selected priority and any subsequent changes.

---

## 4. Incident Documentation Requirements

Every troubleshooting ticket should include:

- A clear summary of the reported issue.
- Affected user, device, application, or business function.
- Date and approximate time the issue started.
- Number of affected users, when known.
- Error messages and relevant diagnostic findings.
- Troubleshooting steps performed and their outcomes.
- Current priority and justification.
- Escalation destination and next action.

Do not include passwords, MFA codes, API keys, access tokens, or unnecessary sensitive information in ticket notes.

---

## 5. Runbook Usage by the IT Support Copilot

The BluePeak Technologies IT Support Copilot should use this runbook to:

- Distinguish individual-user issues from widespread service incidents.
- Recommend safe, ordered troubleshooting steps.
- Identify when escalation is more appropriate than additional local troubleshooting.
- Classify tickets using the P1–P4 definitions.
- Explain the evidence supporting its recommendations.
- Cite the relevant handbook or runbook section.
- Acknowledge uncertainty when the available evidence is insufficient.

The copilot must not invent incident status, claim to have checked a live service dashboard unless an actual integration performed that check, or promise response times that are not documented.

---

## Document Maintenance

**Customer:** BluePeak Technologies  
**Document Owner:** IT Operations  
**Version:** 1.0  
**Status:** Sample project documentation  
**Review Frequency:** To be defined by the customer

Before production use, validate the runbook against the customer's actual service desk procedures, monitoring tools, escalation contacts, and approved service-level agreements.