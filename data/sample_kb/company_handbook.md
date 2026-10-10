# BluePeak Technologies — Internal IT Support Handbook

**Document Type:** Internal IT Knowledge Base  
**Organization:** BluePeak Technologies  
**Department:** Information Technology  
**Audience:** Employees, contractors, and authorized business users  
**Purpose:** Provide consistent guidance for IT support, troubleshooting, account security, and incident reporting.

> **Note:** BluePeak Technologies is a fictional customer used for project development and testing. The policies below are illustrative and should be replaced with verified customer procedures before production use.

## 1. VPN Access and Remote Connectivity

Employees working remotely must use the approved company VPN before accessing internal resources that require a corporate network connection.

### Access Procedure
1. Install the approved VPN client from the company Software Portal.
2. Sign in using your corporate email and password.
3. Complete multi-factor authentication (MFA).
4. Select an approved corporate gateway.
5. Confirm the connection is active before accessing internal systems.

### Troubleshooting
- Verify that your internet connection is working.
- Restart the VPN client and reconnect.
- Check your device's date and time.
- Try another approved gateway.
- If the issue continues, open an IT Service Desk ticket with the error message and a screenshot with sensitive information removed.

**Security rule:** Never disable endpoint protection or firewall controls to bypass a VPN issue.

## 2. Password Management and Account Recovery

- Passwords must contain at least 14 characters.
- Use unique passwords for corporate accounts.
- Use the approved self-service identity portal for routine password resets.
- Accounts may be temporarily locked after five failed login attempts.
- Contact the IT Service Desk if account recovery fails.
- Never share passwords, recovery codes, or MFA codes with anyone, including support personnel.

## 3. Multi-Factor Authentication (MFA)

MFA is required for corporate email, VPN, cloud administration, and other protected systems.

**Approved methods**
- Company-approved authenticator application.
- Registered hardware security key.
- SMS only when explicitly enabled as an approved recovery method.

If you receive an unexpected MFA request, do not approve it. Report it to the IT Service Desk or Security team immediately.

If you replace or lose your registered device, follow the official identity verification and MFA recovery process.

## 4. Software Installation and Application Access

Employees must install business applications through the approved Software Portal or another explicitly authorized distribution channel.

Software not listed in the portal requires manager approval and an IT security review. Requests should include the application name, business purpose, required permissions, and licensing needs.

Administrator access is not automatically granted for installing unapproved applications. Do not bypass application controls or install unauthorized executables.

## 5. Lost or Stolen Company Devices

Immediately report a lost or stolen company laptop or mobile device to the IT Service Desk and your manager.

Provide the device asset tag, last known location, approximate time of loss, and any relevant circumstances.

Authorized IT and Security personnel may lock the device, revoke sessions, initiate a remote wipe, and review relevant security logs.

Do not attempt to recover the device remotely yourself or delay reporting while searching for it.

## 6. Email Security and Phishing

Be cautious of unexpected links, attachments, credential requests, payment instructions, and messages creating unusual urgency.

If you receive a suspicious message:
1. Do not click links or open attachments.
2. Use the approved phishing-reporting feature or contact IT Security.
3. Preserve the original message as directed.
4. If you entered credentials or opened a suspicious attachment, report the incident immediately.

Never send passwords, MFA codes, or confidential company information to an unverified recipient.

## 7. Email and Collaboration Troubleshooting

If corporate email, calendars, or collaboration tools are not working:

- Verify internet connectivity.
- Confirm that the correct corporate account is selected.
- Check whether the problem occurs in both browser and desktop versions.
- Restart the affected application.
- Sign out and sign back in using the approved authentication process.
- Record error messages and when the issue occurred.

Do not forward confidential company messages to personal email accounts as a workaround.

## 8. Network, Wi-Fi, and Printer Support

### Corporate Wi-Fi
Verify that Wi-Fi is enabled and connect using the approved corporate network. If authentication fails, reconnect and record the error message. Contact IT if the problem continues.

### Printers
Confirm the selected printer, check for device errors, and verify network connectivity. Clear stuck print jobs using standard operating-system controls. Escalate persistent problems to the IT Service Desk.

Do not bypass network restrictions or access administrative settings without authorization.

## 9. Access Requests and Employee Onboarding

Access is granted according to the principle of least privilege: employees receive only the permissions required for their approved responsibilities.

Requests should include the employee's corporate identifier, requested system, business justification, required access level, and applicable manager or data-owner approval.

Access may require additional Security approval. Do not use another employee's account while waiting for access.

## 10. Security Incidents and Suspicious Activity

Report suspected account compromise, malware, unauthorized access, data exposure, unexpected login alerts, and lost devices promptly.

When reporting an incident:
1. Contact the IT Service Desk or Security team.
2. Describe what happened and when.
3. Identify affected devices, accounts, or systems when known.
4. Preserve relevant evidence.
5. Follow authorized containment instructions.

Do not investigate other employees' accounts, delete evidence, or share incident details externally without authorization.

## 11. IT Service Desk Tickets and Escalation

Include the following information when opening a support ticket:

- A clear issue title.
- Affected application or device.
- Date and approximate time the issue started.
- Troubleshooting steps already attempted.
- Error messages or sanitized screenshots.
- Business impact and urgency.

| Priority | Example | Recommended action |
|---|---|---|
| Critical | Widespread outage or major suspected security incident | Use the designated urgent reporting channel |
| High | Business-critical system unavailable to a team | Submit a ticket and notify the appropriate support contact |
| Normal | Individual application or access problem | Submit a standard support ticket |
| Low | General question or routine software request | Submit a non-urgent service request |

These categories are illustrative. Actual priority and response targets depend on BluePeak Technologies' approved support procedures.

Update an existing ticket when possible rather than creating duplicates.

## 12. Data Protection and Safe Information Sharing

- Share only the information needed to resolve the issue.
- Remove passwords, API keys, tokens, and session cookies from logs and screenshots.
- Use approved support and file-sharing channels.
- Do not upload confidential documents to unapproved external AI services.
- Verify recipients before sharing sensitive information.
- Follow applicable data-classification and retention requirements.

If sensitive information is exposed, promptly notify the appropriate IT or Security contact. Describe the exposure without reproducing the secret in the ticket.

## 13. Knowledge Base and AI Support Guidelines

The BluePeak IT Support Copilot uses approved knowledge-base documents to answer employee questions and recommend troubleshooting steps.

The system should:
- Ground answers in relevant internal documentation.
- Provide source citations where available.
- Use approved web search only when configured and appropriate.
- Clearly identify missing or conflicting evidence.
- Avoid inventing policies, approval decisions, or response-time guarantees.
- Escalate security incidents and unresolved issues to the IT Service Desk.
- Never request passwords, MFA codes, API keys, or other secrets from users.

## Document Maintenance

**Customer:** BluePeak Technologies  
**Document Owner:** IT Operations  
**Version:** 1.0  
**Status:** Sample project documentation  
**Review Frequency:** To be defined by the customer

Production deployment requires validation against BluePeak Technologies' actual IT policies, approved tools, support contacts, and security requirements.