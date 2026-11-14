"""Held-out real-world check: hand-written modern emails NOT produced by the generator and NOT in any dataset.

Each case can carry an authentication result ("pass" / "fail" / None), as a real mailbox would provide through the
Authentication-Results header. Cases with auth=None test the text model alone (e.g. web UI paste).

  python scripts/realworld_eval.py [classic|transformer|ensemble ...]
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishguard.config import REPORTS_DIR  # noqa: E402
from phishguard.detector import PhishingDetector  # noqa: E402

P, L = "phishing", "legitimate"
CASES = [
    # expected, auth, sender, subject, body
    (P, None, '"Microsoft 365" <no-reply@m365-securemail.com>', "Action required: password expires today",
     "Your Office 365 password expires today. To keep using your current password, confirm it below.\nKeep current password: https://m365-securemail.com/auth/keep?u=you@company.com\nMicrosoft 365 Admin"),
    (P, None, '"DocuSign" <dse@docusign-notify.net>', "You have received a document to review and sign",
     "Please review the attached Payment Remittance document. REVIEW DOCUMENT: https://bit.ly/3xYzAbc\nThis link expires in 48 hours."),
    (P, None, '"HDFC Bank" <alerts@hdfc-kyc-update.in>', "KYC update pending - account will be blocked",
     "Dear Customer, your account will be blocked today due to pending KYC. Update PAN details immediately: http://hdfc-kyc-update.in/login"),
    (P, None, '"CEO" <ceo.office@gmail.com>', "Quick favour",
     "Are you at your desk? I need you to buy 5 Amazon gift cards of $200 each for a client. Scratch the codes and send me pictures. Keep this confidential, I'm in a meeting."),
    (P, None, '"Netflix" <info@netflix-billing-support.com>', "Your membership is on hold",
     "We were unable to process your payment. Update your billing information within 24 hours to avoid cancellation: http://netflix-billing-support.com/update"),
    (P, "pass", '"IT Helpdesk" <helpdesk@it-servicedesk-portal.com>', "Mandatory: re-validate your mailbox",
     "Hello,\nOur records show your mailbox has not been re-validated for the new security policy. Staff who do not complete validation will lose email access on Monday.\nValidate here: https://it-servicedesk-portal.com/owa/validate\nIT Service Desk"),
    (P, "fail", '"Amazon" <account-update@amazon.com>', "There's a problem with your order",
     "Hello,\nWe couldn't charge your payment method for your recent order and it has been put on hold. Please confirm your billing address and card information so we can ship your items.\nConfirm details: http://amzn-orders-help.top/confirm\nAmazon Customer Service"),
    (P, None, '"Finance Dept" <accounts@globalfreight-ltd.co>', "Updated remittance details",
     "Hi,\nPlease note that our bank details have changed due to an audit. Kindly make all future payments, including invoice INV-20931 for $18,450, to the new account in the attached form. Confirm by reply once payment is scheduled.\nThanks,\nMaria, Accounts Receivable"),
    (P, None, '"Instagram" <security@lnstagram-help.com>', "Copyright infringement on your account",
     "Your account has been reported for copyright infringement. If you think this is a mistake, you must submit an appeal form within 24 hours or your account will be permanently deleted. Appeal: https://lnstagram-help.com/appeal"),
    (P, None, '"Recruiter" <hr.amazonjobs@outlook.com>', "Remote data entry position - $450/day",
     "Congratulations, your profile was shortlisted for a remote position paying $450 per day. To proceed, pay the refundable onboarding kit fee of $99 via the link and send a copy of your Aadhaar and bank passbook. https://tinyurl.com/remote-onboard"),
    (P, None, '"Wallet Support" <support@metamask-secure.app>', "Your wallet will be deactivated",
     "Due to a new regulation, unverified wallets will be deactivated. Verify ownership by entering your 12-word secret recovery phrase: https://metamask-secure.app/verify"),
    (P, None, '"SharePoint" <noreply@sharepoint-files.online>', "Rakesh shared 'Q4 Bonus Allocation.xlsx' with you",
     "Rakesh Kumar has shared a file with you.\nQ4 Bonus Allocation.xlsx\nOpen: https://sharepoint-files.online/view?doc=8812\nYou will be asked to sign in with your work email and password."),
    (P, None, '"Delhivery" <track@delhivery-redelivery.info>', "Package held at warehouse",
     "Your package could not be delivered because of an unpaid customs fee of Rs 49. Pay within 12 hours or it will be returned to sender: http://delhivery-redelivery.info/pay"),
    (P, None, '"Apple" <id@apple-support-center.net>', "Your Apple ID has been locked",
     "Your Apple ID has been locked for security reasons. Someone tried to sign in from an unknown device. Unlock your account now: http://182.16.44.9/appleid/unlock"),
    (P, None, '"Income Tax Dept" <refund@incometax-refund.org>', "Refund of Rs 15,490 approved",
     "Dear Taxpayer, you are eligible for a tax refund of Rs 15,490. Submit your debit card number, expiry and CVV to receive the refund in your account: http://incometax-refund.org/claim"),
    (P, None, '"Zoom" <meetings@zoom-invite.co>', "Missed meeting - recording available",
     "You missed a scheduled meeting. Sign in with your corporate email to watch the recording before it is deleted: https://zoom-invite.co/rec/login"),
    (L, "pass", '"Amazon.in" <order-update@amazon.in>', "Your Amazon.in order #403-1234567-1234567 has shipped",
     "Hello Ravi, your package with Boat Airdopes 141 has shipped and will arrive Thursday, 19 November. Track your package in Your Orders. Order total: Rs. 1,299.00"),
    (L, None, '"Amazon.in" <order-update@amazon.in>', "Your Amazon.in order #403-1234567-1234567 has shipped",
     "Hello Ravi, your package with Boat Airdopes 141 has shipped and will arrive Thursday, 19 November. Track your package in Your Orders. Order total: Rs. 1,299.00"),
    (L, "pass", '"GitHub" <noreply@github.com>', "[GitHub] A new SSH key was added to your account",
     "Hey octocat! A new public key was added to your account. If you did not add this key, please remove it from your account settings. Thanks, The GitHub Team"),
    (L, "pass", '"Google" <no-reply@accounts.google.com>', "Security alert",
     "New sign-in to your Google Account on a Windows device. If this was you, you don't need to do anything. If not, we'll help you secure your account. Check activity at https://myaccount.google.com/notifications"),
    (L, None, '"Google" <no-reply@accounts.google.com>', "Security alert",
     "New sign-in to your Google Account on a Windows device. If this was you, you don't need to do anything. If not, we'll help you secure your account. Check activity at https://myaccount.google.com/notifications"),
    (L, None, '"Ankit Verma" <ankit.verma@infosys.com>', "Re: Interview schedule for Thursday",
     "Hi Arjun, thanks for confirming. The technical round is on Thursday at 11 AM IST on Teams; I'll send the invite shortly. Please keep your resume and ID handy. Regards, Ankit"),
    (L, "pass", '"Swiggy" <noreply@swiggy.in>', "Your Swiggy order receipt",
     "Thanks for ordering with Swiggy! Order ID 1489123456 from Meghana Foods. Item total Rs 540, delivery fee Rs 25, paid via UPI. Rate your order in the app."),
    (L, None, '"Coursera" <no-reply@m.mail.coursera.org>', "Your weekly learning summary",
     "You completed 3 videos and 1 quiz in Machine Learning Specialization this week. Keep it up! Next: Week 4 - Neural Networks. Unsubscribe or manage email preferences."),
    (L, "pass", '"HDFC Bank" <alerts@hdfcbank.net>', "Transaction alert",
     "Dear Customer, Rs.2,500.00 has been debited from account **4321 to VPA swiggy@icici on 14-11-26. Your UPI transaction reference number is 431812345678. Not you? Call 18002586161."),
    (L, None, '"HDFC Bank" <alerts@hdfcbank.net>', "Transaction alert",
     "Dear Customer, Rs.2,500.00 has been debited from account **4321 to VPA swiggy@icici on 14-11-26. Your UPI transaction reference number is 431812345678. Not you? Call 18002586161."),
    (L, None, '"Mom" <sunita.mohanty62@gmail.com>', "Diwali plans",
     "Beta, when are you coming home for Diwali? Papa wants to book the train tickets this week. Also bring the charger you borrowed last time. Call me tonight."),
    (L, None, '"Neha Gupta" <neha.g@acme-corp.com>', "Updated slides for Monday",
     "Hey, I've uploaded the updated deck to the team drive with the new revenue numbers from finance. Can you review slides 8-12 before Monday's client call? Thanks!"),
    (L, "pass", '"Zoom" <no-reply@zoom.us>', "Cloud recording - Weekly Sync is now available",
     "Hi Arjun, your cloud recording is now available. Topic: Weekly Sync. Date: Nov 12, 2026. You can view the recording from your Zoom web portal under Recordings."),
    (L, None, '"IRCTC" <ticketadmin@irctc.co.in>', "Booking Confirmation PNR 4512378890",
     "Your ticket has been booked. Train 12802 Purushottam Express, Bhubaneswar to New Delhi, Date of Journey 20-12-2026, Coach B3 Berth 34. Thank you for using IRCTC."),
    (L, "pass", '"LinkedIn" <invitations@linkedin.com>', "Rohan Mehta wants to connect",
     "Rohan Mehta, Senior Data Scientist at Flipkart, wants to connect with you. Accept or ignore the invitation on LinkedIn."),
    (L, None, '"Prof. S. Rao" <srao@iitbbs.ac.in>', "Project report deadline extended",
     "Dear students, the submission deadline for the final project report has been extended to 28 November. Please upload a single PDF on the course portal. Late submissions will not be accepted after that."),
]


def build_headers(auth: str | None, sender: str) -> dict:
    if not auth:
        return {}
    dom = sender.rsplit("@", 1)[-1].rstrip(">")
    if auth == "pass":
        return {"Authentication-Results": f"mx.google.com; dkim=pass header.d={dom}; spf=pass smtp.mailfrom={dom}; dmarc=pass header.from={dom}"}
    return {"Authentication-Results": f"mx.google.com; dkim=fail; spf=fail smtp.mailfrom={dom}; dmarc=fail header.from={dom}"}


def run(model: str) -> dict:
    det = PhishingDetector(model_choice=model)
    ok = 0
    rows = []
    print(f"\n=== model: {det.model_used} ===")
    for exp, auth, sender, subj, body in CASES:
        v = det.predict_fields(subj, body, sender, headers=build_headers(auth, sender))
        good = v.label == exp
        ok += good
        rows.append({"expected": exp, "got": v.label, "score": v.score, "ml": v.model_score, "auth": auth, "subject": subj})
        print(f"{'OK ' if good else 'XX '} exp={exp:10s} got={v.label:10s} score={v.score:.2f} ml={v.model_score:.2f} "
              f"auth={str(auth):4s} {subj[:50]}")
    tp = sum(r["expected"] == P and r["got"] == P for r in rows)
    fp = sum(r["expected"] == L and r["got"] == P for r in rows)
    n_p, n_l = sum(r["expected"] == P for r in rows), sum(r["expected"] == L for r in rows)
    summary = {"model": det.model_used, "correct": ok, "total": len(rows), "phishing_caught": f"{tp}/{n_p}",
               "legit_false_alarms": f"{fp}/{n_l}", "cases": rows}
    print(f"{ok}/{len(rows)} correct | phishing caught {tp}/{n_p} | false alarms on legitimate {fp}/{n_l}")
    return summary


if __name__ == "__main__":
    results = [run(m) for m in (sys.argv[1:] or ["classic"])]
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "realworld_eval.json").write_text(json.dumps(results, indent=2))
