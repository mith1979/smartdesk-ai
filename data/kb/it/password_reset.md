---
id: it-password-001
company: NovaTech Solutions
domain: IT
topic: password_reset
version: "1.0"
source_type: synthetic
review_status: reviewed
---

# NovaTech Solutions — Password Reset Guide

This is a fictional IT guide for the SmartDesk AI capstone.
The portal and email addresses are demonstration values.

## Scope
This guide covers resetting a NovaTech employee account password.
It does not cover personal accounts or local administrator passwords.

## Standard Password Reset
1. Open https://accounts.novatech.example.
2. Select "Forgot password".
3. Enter your NovaTech employee email address.
4. Complete verification using your registered MFA method.
5. Choose a new password and submit the reset.
6. Sign in again using the new password.

The demonstration password requirements are:
- At least 14 characters.
- Must not match any of your previous 5 passwords.

Never share your password or MFA verification codes with SmartDesk
or another employee.

## Forgotten Password with Available MFA
Use the standard password-reset process above.
An existing signed-in session is not required.

## Lost or Unavailable MFA
If you cannot access any registered MFA method, self-service reset
cannot be completed. Contact helpdesk@novatech.example.

IT must verify your identity through its approved recovery process.
SmartDesk cannot bypass MFA or perform identity verification.

## Account Lockout
After 5 consecutive failed password attempts, the account is locked
for 15 minutes.

Wait 15 minutes after the lockout, then try again.
If you do not remember the password, use the reset process.
If access still fails, contact IT.

## Applications Still Using the Old Password
After a successful reset:
1. Sign out of the affected application.
2. Sign in with the new password.
3. Update saved credentials if the application stores them.

Do not repeatedly retry the old password, as this may trigger lockout.

## Escalation
If reset fails, provide IT with:
- Employee email.
- A brief description of the failed step.
- Exact error message, with secrets removed.
- Approximate time of the failure.

Never include passwords, MFA codes, or recovery codes in a ticket.
Contact: helpdesk@novatech.example.

## Limits
This guide does not cover VPN configuration, MFA enrollment,
or local administrator password recovery.