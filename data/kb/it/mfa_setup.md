---
id: it-mfa-001
company: NovaTech Solutions
domain: IT
topic: mfa
version: "1.0"
source_type: synthetic
review_status: reviewed
---

# NovaTech Solutions — MFA Setup and Recovery

This is a fictional IT guide for the SmartDesk AI capstone.
All portal URLs and contacts are demonstration values.

## Purpose
Multi-factor authentication (MFA) adds a verification step
after entering your password.

NovaTech requires MFA for employee accounts.
The supported method is an authenticator application that
generates time-based verification codes.

## First-Time Enrollment
New employees receive an enrollment invitation at their
registered onboarding email address.

1. Follow the invitation to https://accounts.novatech.example.
2. Complete the initial account setup.
3. Select Security > Add authenticator.
4. Scan the displayed QR code using the authenticator application.
5. Enter the generated verification code to confirm enrollment.
6. Save the recovery codes in a secure personal location.

The enrollment invitation expires after 24 hours.
Contact helpdesk@novatech.example if it expires or never arrives.

## Verification Code Not Working
1. Enable automatic date and time on your phone.
2. Confirm you selected the NovaTech account in the authenticator.
3. Wait for a fresh code and enter it before it expires.
4. If verification still fails, contact IT.

Never share verification codes with SmartDesk or another employee.

## Moving to a New Phone
If your old authenticator still works:
1. Sign in to the employee account portal.
2. Open Security > Add authenticator.
3. Enroll the new phone and verify its code.
4. Remove the old authenticator only after the new one works.

## Lost Phone or Unavailable Authenticator
Use a saved recovery code if available.
Each recovery code can be used only once.

After signing in, enroll a replacement authenticator and
generate a new set of recovery codes. Generating a new set
invalidates the previous set.

If neither an authenticator nor recovery codes are available,
contact helpdesk@novatech.example.

IT must verify your identity through its approved recovery process.
SmartDesk cannot disable MFA or bypass identity verification.

## Unexpected Verification Requests
Do not approve authentication requests you did not initiate.
Contact IT to report suspected unauthorized access.

## Information for an IT Ticket
Include:
- Employee email.
- Whether this is enrollment, code failure, or device recovery.
- Exact error message with secrets removed.

Never include QR codes, setup secrets, verification codes,
recovery codes, or passwords.

## Limits
This guide does not describe IT's identity-verification procedure
or guarantee a recovery completion time.
Refer those questions to IT.