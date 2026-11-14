"""Generate a modern, template-based email set to correct a known bias in the public corpora.

Why: the legitimate class of the public data is mostly 2001-2008 corporate mail (Enron, SpamAssassin, CEAS). It
contains almost no *legitimate automated notifications* (receipts, shipping, OTPs, security alerts, SaaS mail), so a
model trained only on it flags most modern transactional mail as phishing.

Design rules that stop the model from learning shortcuts instead of real phishing signals:
  * the SAME brands appear in both classes (brand name is not a signal);
  * footers, greetings, formatting, length and politeness are randomised in BOTH classes;
  * legitimate mail sometimes uses mild urgency ("OTP valid for 10 minutes", "payment due"), and phishing is
    sometimes calm and well-written, so tone alone cannot decide;
  * what actually differs is the intent: phishing asks you to log in / verify / pay / reply with secrets under
    threat, to a non-official destination, while legitimate mail informs and points to official channels.

Output: data/raw/Modern_Synthetic.csv  (subject, body, sender, label, source)
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from phishguard.config import RANDOM_SEED, RAW_DIR  # noqa: E402

R = random.Random(RANDOM_SEED)

BRANDS = {
    # name: (official sending domain, official web domain, category)
    "Amazon": ("amazon.in", "amazon.in", "shop"), "Flipkart": ("flipkart.com", "flipkart.com", "shop"),
    "Myntra": ("myntra.com", "myntra.com", "shop"), "Google": ("accounts.google.com", "myaccount.google.com", "tech"),
    "Microsoft": ("accountprotection.microsoft.com", "account.microsoft.com", "tech"),
    "Apple": ("id.apple.com", "appleid.apple.com", "tech"), "Netflix": ("mailer.netflix.com", "netflix.com", "sub"),
    "Spotify": ("spotify.com", "spotify.com", "sub"), "PayPal": ("paypal.com", "paypal.com", "bank"),
    "HDFC Bank": ("hdfcbank.net", "hdfcbank.com", "bank"), "SBI": ("sbi.co.in", "onlinesbi.sbi", "bank"),
    "ICICI Bank": ("icicibank.com", "icicibank.com", "bank"), "Axis Bank": ("axisbank.com", "axisbank.com", "bank"),
    "Paytm": ("paytm.com", "paytm.com", "bank"), "GitHub": ("github.com", "github.com", "dev"),
    "GitLab": ("gitlab.com", "gitlab.com", "dev"), "LinkedIn": ("linkedin.com", "linkedin.com", "social"),
    "Instagram": ("mail.instagram.com", "instagram.com", "social"), "Facebook": ("facebookmail.com", "facebook.com", "social"),
    "Swiggy": ("swiggy.in", "swiggy.com", "food"), "Zomato": ("zomato.com", "zomato.com", "food"),
    "Uber": ("uber.com", "uber.com", "ride"), "Ola": ("olacabs.com", "olacabs.com", "ride"),
    "DHL": ("dhl.com", "dhl.com", "ship"), "FedEx": ("fedex.com", "fedex.com", "ship"),
    "Blue Dart": ("bluedart.com", "bluedart.com", "ship"), "India Post": ("indiapost.gov.in", "indiapost.gov.in", "ship"),
    "Dropbox": ("dropbox.com", "dropbox.com", "tech"), "Zoom": ("zoom.us", "zoom.us", "tech"),
    "Slack": ("slack.com", "slack.com", "tech"), "DocuSign": ("docusign.net", "docusign.com", "tech"),
    "Adobe": ("adobe.com", "adobe.com", "tech"), "Airtel": ("airtel.in", "airtel.in", "telco"),
    "Jio": ("jio.com", "jio.com", "telco"), "IRCTC": ("irctc.co.in", "irctc.co.in", "travel"),
    "MakeMyTrip": ("makemytrip.com", "makemytrip.com", "travel"), "Coursera": ("m.mail.coursera.org", "coursera.org", "edu"),
    "Udemy": ("udemy.com", "udemy.com", "edu"), "Income Tax Department": ("incometax.gov.in", "incometax.gov.in", "gov"),
    "Coinbase": ("coinbase.com", "coinbase.com", "bank"), "Office 365": ("microsoft.com", "office.com", "tech"),
}
FIRST = ["Aarav", "Priya", "Rahul", "Ananya", "Vikram", "Sneha", "Arjun", "Kavya", "Rohan", "Meera", "Aditya", "Isha",
         "Karan", "Divya", "Sanjay", "Pooja", "Amit", "Neha", "Rajesh", "Sunita", "John", "Emily", "Michael", "Sarah",
         "David", "Laura", "Chen", "Fatima", "Omar", "Sofia", "Lucas", "Mia", "Ravi", "Lakshmi", "Suresh", "Nisha"]
LAST = ["Sharma", "Patel", "Iyer", "Reddy", "Nair", "Gupta", "Singh", "Das", "Mehta", "Kapoor", "Rao", "Mishra",
        "Smith", "Johnson", "Brown", "Lee", "Garcia", "Khan", "Mohanty", "Pillai", "Bose", "Joshi", "Verma", "Kumar"]
ITEMS = ["Boat Airdopes 141", "Samsung 25W charger", "Prestige pressure cooker", "Nike running shoes", "Kindle Paperwhite",
         "HP wireless mouse", "Milton water bottle", "The Psychology of Money (paperback)", "Logitech K380 keyboard",
         "Philips trimmer", "USB-C cable 2m", "Yoga mat", "Mi Power Bank 20000mAh", "Levi's denim jacket", "Desk lamp"]
CITIES = ["Bengaluru", "Mumbai", "Pune", "Hyderabad", "Chennai", "Delhi", "Kolkata", "Bhubaneswar", "London", "Austin"]
DEVICES = ["Chrome on Windows", "Safari on iPhone", "Firefox on Ubuntu", "Edge on Windows 11", "Android app", "Mac OS X"]
CURR = ["Rs.", "₹", "INR", "$", "USD"]
COMPANIES = ["Acme Corp", "Nimbus Labs", "Tata Elxsi", "BlueOrbit Systems", "Infosys", "Wipro", "Zeta Analytics", "Bright Retail"]


def name() -> str:
    return f"{R.choice(FIRST)} {R.choice(LAST)}"


def amount() -> str:
    return f"{R.choice(CURR)} {R.randint(99, 45000):,}.{R.choice(['00', '50', '99'])}"


def oid(n: int = 10) -> str:
    return "".join(R.choice("0123456789") for _ in range(n))


def date() -> str:
    return f"{R.randint(1, 28)} {R.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} 202{R.randint(4, 6)}"


def greet(person: str, generic_prob: float) -> str:
    if R.random() < generic_prob:
        return R.choice(["Dear Customer,", "Dear User,", "Hello,", "Dear Valued Customer,", "Hi there,", "Dear Account Holder,", "Dear Member,"])
    first = person.split()[0]
    return R.choice([f"Hi {first},", f"Hello {first},", f"Dear {person},", f"Dear {first},", f"Hey {first},", f"{first},"])


def footer(brand: str) -> str:
    if R.random() < 0.35:
        return ""
    return R.choice([
        f"\n\n© 202{R.randint(4, 6)} {brand}. All rights reserved.",
        "\n\nThis is an automated message, please do not reply.",
        f"\n\nYou are receiving this email because you have an account with {brand}. Privacy Policy | Terms of Use",
        "\n\nUnsubscribe | Manage email preferences | Help Centre",
        f"\n\n{brand} Customer Care | Privacy | Terms",
    ])


def sign(brand: str) -> str:
    return R.choice([f"\n\nThanks,\nThe {brand} Team", f"\n\nRegards,\n{brand}", f"\n\n- {brand} Support",
                     f"\n\nWarm regards,\nTeam {brand}", "", f"\n\nSincerely,\n{brand} Security"])


# ------------------------------------------------------------------ legitimate
def legit(brand: str) -> tuple[str, str, str]:
    send, web, cat = BRANDS[brand]
    p = name()
    g = greet(p, 0.15)
    link = f"https://{web}/{R.choice(['account', 'orders', 'help', 'settings', 'activity', 'billing'])}"
    kind = R.choice({
        "shop": ["order", "shipped", "delivered", "refund", "otp", "newsletter"],
        "tech": ["signin", "newdevice_ok", "password_changed", "storage", "invoice", "share", "otp", "meeting"],
        "sub": ["renewal", "receipt", "newsletter", "signin"],
        "bank": ["debit", "credit", "statement", "otp", "signin", "emi"],
        "dev": ["sshkey", "pr", "ci", "signin", "security_advisory"],
        "social": ["connection", "signin", "digest", "otp"],
        "food": ["receipt", "refund", "offer"], "ride": ["trip", "receipt"],
        "ship": ["tracking", "delivered"], "telco": ["bill", "recharge"], "travel": ["ticket", "booking"],
        "edu": ["digest", "certificate", "receipt"], "gov": ["itr_ack", "refund_processed"],
    }[cat])
    item, amt, ref, d = R.choice(ITEMS), amount(), oid(), date()
    T = {
        "order": (f"Your {brand} order #{oid(3)}-{oid(7)} is confirmed",
                  f"{g}\n\nThank you for your order. We'll let you know when {item} ships.\nOrder total: {amt}\nEstimated delivery: {d}\n\nYou can view or manage your order in Your Orders: {link}"),
        "shipped": (R.choice([f"Shipped: {item}", f"Your package is on the way", f"Your {brand} order has shipped"]),
                    f"{g}\n\nGood news! Your package containing {item} has shipped and will arrive by {d}.\nTracking ID: {oid(12)}\nTrack your package: {link}"),
        "delivered": (f"Delivered: your {brand} package", f"{g}\n\nYour package was delivered on {d}. It was handed to a resident at {R.choice(CITIES)}.\nHow was your delivery? Rate it in the app."),
        "refund": (f"Your refund of {amt} has been processed", f"{g}\n\nWe've processed a refund of {amt} for order {ref}. It should reflect in your original payment method within 5-7 business days. No action is needed from you."),
        "otp": (R.choice([f"{brand} verification code", f"Your one-time password", f"{oid(6)} is your {brand} code"]),
                f"{g}\n\nYour verification code is {oid(6)}. It is valid for {R.choice([5, 10, 15])} minutes.\nNever share this code with anyone. {brand} will never call or email you asking for it.\nIf you didn't request this, you can ignore this email."),
        "newsletter": (R.choice([f"New arrivals picked for you", f"This week's top deals at {brand}", f"{brand} Big Sale starts {d}"]),
                       f"{g}\n\nUp to {R.randint(10, 70)}% off on {item} and more. Sale runs till {d}. Browse the collection at https://{web}/deals"),
        "signin": (R.choice(["Security alert", f"New sign-in to your {brand} account", "New login detected"]),
                   f"{g}\n\nWe noticed a new sign-in to your {brand} account from {R.choice(DEVICES)} near {R.choice(CITIES)} on {d}.\nIf this was you, you don't need to do anything. If not, review your account activity by opening the {brand} app or visiting {web} directly."),
        "newdevice_ok": (f"You signed in on a new device", f"{g}\n\nYour {brand} account was just used to sign in on {R.choice(DEVICES)}. If this was you, you can ignore this message. Otherwise go to {web}/security to secure your account."),
        "password_changed": (f"Your {brand} password was changed", f"{g}\n\nThe password for your {brand} account was changed on {d}. If you made this change, no further action is required. If you didn't, contact support through {web}."),
        "storage": (f"Your {brand} storage is {R.randint(80, 99)}% full", f"{g}\n\nYou're running low on storage. You can free up space or upgrade your plan anytime from {link}."),
        "invoice": (f"Your {brand} invoice for {d}", f"{g}\n\nYour invoice {ref} for {amt} is available. Payment was charged to your card ending {oid(4)}. View invoice: {link}"),
        "share": (f"{name()} shared a file with you", f"{g}\n\n{name()} shared \"{R.choice(['Q3 report', 'Project plan', 'Team photos', 'Budget 2026'])}\" with you on {brand}. Open it from your {brand} home page."),
        "meeting": (f"Invitation: {R.choice(['Sprint review', 'Weekly sync', '1:1', 'Design review'])} @ {d}", f"{g}\n\n{name()} has invited you to a meeting on {d} at {R.randint(9, 18)}:00 IST.\nJoin from the calendar invite. Accept | Decline | Maybe"),
        "renewal": (f"Your {brand} membership renews soon", f"{g}\n\nYour plan renews on {d} for {amt}. No action is required. To change your plan, go to Account settings."),
        "receipt": (R.choice([f"Your {brand} receipt", f"Payment received - thank you", f"Receipt for order {ref}"]),
                    f"{g}\n\nThanks for your payment of {amt} on {d}.\nTransaction ID: {ref}\nPaid using {R.choice(['UPI', 'card ending ' + oid(4), 'wallet', 'net banking'])}."),
        "debit": ("Transaction alert", f"{g}\n\n{amt} has been debited from your account **{oid(4)} on {d} towards {R.choice(['UPI/' + oid(12), 'POS ' + R.choice(['AMAZON', 'SWIGGY', 'DMART']), 'NEFT'])}. Available balance: {amount()}.\nIf you did not do this transaction, call our 24x7 helpline printed on the back of your card."),
        "credit": ("Amount credited to your account", f"{g}\n\n{amt} has been credited to your account **{oid(4)} on {d}. Ref no {ref}."),
        "statement": (f"Your {brand} statement for {d} is ready", f"{g}\n\nYour account statement is ready. Total due: {amt}, due date {d}. You can view it in net banking or the mobile app. Please ignore if already paid."),
        "emi": (f"EMI reminder: {amt} due on {d}", f"{g}\n\nThis is a reminder that your EMI of {amt} for loan account {ref} is due on {d}. Please maintain sufficient balance. Ignore if already paid."),
        "sshkey": (f"[{brand}] A new SSH key was added to your account", f"{g}\n\nA new public key was added to your account. If you did not add this key, remove it from your SSH keys settings."),
        "pr": (f"[{R.choice(['api', 'web', 'infra'])}-service] PR #{R.randint(10, 999)} merged", f"{name()} merged pull request #{R.randint(10, 999)}: {R.choice(['Fix login redirect', 'Add caching layer', 'Bump dependencies', 'Refactor parser'])}.\nView it on {brand}: https://{web}/org/repo/pull/{R.randint(10, 999)}"),
        "ci": (f"[{brand}] Run failed: build - main", f"The workflow 'build' failed on branch main ({oid(7)}). {R.randint(1, 9)} tests failed. View the run logs for details."),
        "security_advisory": (f"[{brand}] Dependabot alert in your repository", f"A {R.choice(['high', 'moderate'])} severity vulnerability was found in a dependency of your repository. Review the advisory and update the package version."),
        "connection": (f"{name()} wants to connect", f"{g}\n\n{name()}, {R.choice(['Software Engineer', 'Recruiter', 'Data Scientist'])} at {R.choice(COMPANIES)}, sent you an invitation. Accept | Ignore"),
        "digest": (R.choice([f"Your weekly summary", f"You have {R.randint(2, 30)} new notifications", "Your learning progress this week"]),
                   f"{g}\n\nHere's what you missed this week: {R.randint(2, 9)} new posts, {R.randint(1, 5)} comments and {R.randint(1, 20)} likes. Keep going!"),
        "trip": (f"Your trip with {brand} on {d}", f"{g}\n\nThanks for riding with {brand}. Trip from {R.choice(CITIES)} Airport, fare {amt}, paid via {R.choice(['UPI', 'cash', 'card'])}. Rate your driver in the app."),
        "offer": (f"{R.randint(20, 60)}% off on your next order", f"{g}\n\nUse code SAVE{R.randint(10, 99)} to get {R.randint(20, 60)}% off on orders above {amount()}. Valid till {d}. T&C apply."),
        "tracking": (f"Shipment {oid(10)} is out for delivery", f"{g}\n\nYour shipment from {R.choice(COMPANIES)} is out for delivery today and will reach you by 8 PM. Track at https://{web}/track"),
        "bill": (f"Your {brand} bill for {d}", f"{g}\n\nYour bill of {amt} is generated. Due date: {d}. Pay using the {brand} app or any UPI app. Ignore if already paid."),
        "recharge": ("Recharge successful", f"{g}\n\nYour recharge of {amt} is successful. Plan valid till {d}. Transaction ID {ref}."),
        "ticket": (f"Booking confirmed - PNR {oid(10)}", f"{g}\n\nYour ticket is booked for {d}. Train {R.randint(10000, 22999)}, {R.choice(CITIES)} to {R.choice(CITIES)}. Coach B{R.randint(1, 9)}, seat {R.randint(1, 72)}. Happy journey!"),
        "booking": (f"Your booking {ref} is confirmed", f"{g}\n\nYour hotel in {R.choice(CITIES)} is confirmed for {d}. Total paid {amt}. Manage your booking in My Trips."),
        "certificate": (f"Congratulations! Your certificate is ready", f"{g}\n\nYou completed {R.choice(['Machine Learning', 'Python for Everybody', 'Cloud Fundamentals'])}. Your certificate is available on your profile."),
        "itr_ack": ("ITR filed successfully - acknowledgement", f"{g}\n\nYour Income Tax Return for AY 2026-27 has been successfully filed. Acknowledgement number {oid(15)}. Please e-verify within 30 days through the e-filing portal."),
        "refund_processed": ("Income tax refund processed", f"{g}\n\nYour refund of {amt} for AY 2025-26 has been processed and will be credited to your pre-validated bank account. No action required."),
    }
    subject, body = T[kind]
    return subject, body + sign(brand) + footer(brand), f'"{brand}" <{R.choice(["no-reply", "noreply", "alerts", "info", "notifications"])}@{send}>'


# ------------------------------------------------------------------ phishing
def lookalike(brand: str) -> str:
    b = brand.lower().replace(" ", "")
    tld = R.choice(["com", "net", "xyz", "top", "info", "online", "support", "co", "in", "live", "site", "app"])
    return R.choice([
        f"{b}-account-verify.{tld}", f"secure-{b}-login.{tld}", f"{b}.security-check.{tld}", f"{b}support-help.{tld}",
        f"{b}-kyc-update.{tld}", f"my{b}-billing.{tld}", f"{b.replace('o', '0', 1)}.{tld}", f"{b}-id-recovery.{tld}",
        f"{R.randint(11, 223)}.{R.randint(0, 255)}.{R.randint(0, 255)}.{R.randint(1, 254)}",
        R.choice(["bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "t.ly"]),
        f"{R.choice(['docs-share', 'secure-portal', 'auth-session', 'verify-now', 'webmail-update'])}.{tld}",
    ])


def phish(brand: str) -> tuple[str, str, str]:
    p = name()
    g = greet(p, 0.6)
    dom = lookalike(brand)
    path = R.choice(["login", "verify", "secure/update", "account/restore", "kyc", "billing/update", "auth", "signin.php", "x/" + oid(6)])
    link = f"{R.choice(['http', 'https'])}://{dom}/{path}"
    calm = R.random() < 0.3  # well-written, low-pressure phishing
    deadline = R.choice(["within 24 hours", "within 48 hours", "today", "immediately", "by end of day", "within 12 hours"])
    amt, ref, d = amount(), oid(), date()
    kind = R.choice(["suspend", "verify", "password_expiry", "payment_failed", "kyc", "delivery_fee", "invoice",
                     "docshare", "prize", "refund_claim", "bec_giftcard", "bec_wire", "signin_phish", "mailbox_full",
                     "tax_refund", "crypto", "job_offer", "voicemail", "payroll"])
    pressure = "" if calm else R.choice([
        f" Failure to do so {deadline} will result in permanent suspension of your account.",
        f" If you do not respond {deadline}, your account will be locked.", " This is your final notice.",
        f" Your access will be restricted {deadline}.", " Ignoring this message may lead to legal action.",
    ])
    T = {
        "suspend": (R.choice([f"Your {brand} account has been suspended", f"Account suspended - action required", f"Unusual activity detected on your {brand} account"]),
                    f"{g}\n\nWe detected unusual activity and have temporarily suspended your {brand} account for your protection. To restore access, please confirm your identity: {link}.{pressure}"),
        "verify": (R.choice([f"Verify your {brand} account", f"Action required: confirm your account details", f"Please confirm your information"]),
                   f"{g}\n\nAs part of our routine security update, we need you to verify your account information. Please sign in and confirm your details using the secure link below.\n\n{link}\n{pressure}"),
        "password_expiry": (R.choice([f"Your {brand} password expires today", "Password expiration notice", "Keep your current password"]),
                            f"{g}\n\nYour password is set to expire {R.choice(['today', 'in 1 day', 'shortly'])}. To keep using your current password, confirm it here: {link}\n{pressure}"),
        "payment_failed": (R.choice([f"Your {brand} payment was declined", "Your membership is on hold", "Billing problem with your account"]),
                           f"{g}\n\nWe were unable to process your last payment of {amt}. Please update your card details to avoid interruption of service: {link}.{pressure}"),
        "kyc": (R.choice([f"{brand}: KYC update pending", "Your account will be blocked today", f"Complete your KYC to continue using {brand}"]),
                f"{g}\n\nYour KYC documents have expired. Update your PAN and Aadhaar details now to avoid blocking of your account: {link}.{pressure}"),
        "delivery_fee": (R.choice([f"{brand}: your parcel is on hold", "Delivery attempt failed", f"Action needed for shipment {oid(10)}"]),
                         f"{g}\n\nWe tried to deliver your parcel but the address was incomplete. Pay the redelivery fee of {R.choice(CURR)} {R.randint(25, 150)} to reschedule: {link}.{pressure}"),
        "invoice": (R.choice([f"Invoice {ref} attached", "Overdue payment reminder", f"Payment request from {R.choice(COMPANIES)}"]),
                    f"{g}\n\nPlease find the outstanding invoice of {amt} for services rendered. Kindly process the payment to the updated bank account details in the link: {link}.{pressure}"),
        "docshare": (R.choice([f"{name()} shared a document with you", "You have received a secure document", "Please review and sign"]),
                     f"{g}\n\nA document \"{R.choice(['Payment Remittance', 'Salary Revision 2026', 'Contract Agreement', 'Invoice copy'])}\" has been shared with you via {brand}. Sign in with your email credentials to view: {link}\n{pressure}"),
        "prize": (R.choice(["Congratulations! You have won", "You are our lucky winner", f"{brand} anniversary reward"]),
                  f"{g}\n\nYou have been selected to receive {R.choice(['an iPhone 16', amt, 'a gift voucher worth ' + amt])}. To claim your reward, provide your details and pay a small processing fee here: {link}.{pressure}"),
        "refund_claim": (R.choice([f"You are eligible for a refund of {amt}", "Refund pending - confirm your bank details"]),
                         f"{g}\n\nAfter a review of your account, you are eligible for a refund of {amt}. Enter your card or bank details to receive the refund: {link}.{pressure}"),
        "bec_giftcard": (R.choice(["Quick favour", "Are you available?", "Urgent request", "Need your help"]),
                         f"{g}\n\nAre you at your desk? I need you to buy {R.randint(3, 10)} gift cards of {amount()} each for a client. Scratch the codes and send me pictures. Keep this confidential, I am in a meeting.\n\n{name()}\nCEO"),
        "bec_wire": (R.choice(["Wire transfer needed today", "Confidential payment", "Vendor payment update"]),
                     f"{g}\n\nPlease process a wire transfer of {amt} to the account below for an acquisition we are closing today. This is confidential, do not discuss with anyone. Reply once done.\n\nRegards,\n{name()}"),
        "signin_phish": (R.choice(["Security alert", f"New sign-in to your {brand} account", "Someone has your password"]),
                         f"{g}\n\nWe noticed a sign-in attempt from {R.choice(['Russia', 'Nigeria', 'an unknown device', 'Brazil'])}. If this wasn't you, secure your account now by verifying your password: {link}.{pressure}"),
        "mailbox_full": (R.choice(["Your mailbox is almost full", "Email quota exceeded", "Incoming messages on hold"]),
                         f"{g}\n\nYour mailbox has exceeded its storage limit and {R.randint(5, 40)} incoming messages are on hold. Log in to upgrade your quota and release your messages: {link}.{pressure}"),
        "tax_refund": (R.choice(["Income tax refund approved", "You have a pending tax refund"]),
                       f"{g}\n\nYou are eligible for an income tax refund of {amt}. Submit your bank account and debit card details to receive it: {link}.{pressure}"),
        "crypto": (R.choice(["Your wallet has been compromised", f"{brand}: withdrawal request", "Verify your wallet"]),
                   f"{g}\n\nA withdrawal of {R.randint(1, 9)}.{oid(2)} BTC was requested from your account. If this wasn't you, connect your wallet and enter your recovery phrase to cancel: {link}.{pressure}"),
        "job_offer": (R.choice(["Work from home opportunity", "You're hired!", "Part-time job offer"]),
                      f"{g}\n\nEarn {amount()} per day working from home. No experience needed. Register and pay the one-time registration fee to start: {link}"),
        "voicemail": (R.choice(["New voicemail message", "You have 1 missed voice message"]),
                      f"{g}\n\nYou received a new voicemail ({R.randint(0, 2)}:{R.randint(10, 59)} min). Sign in with your email account to listen: {link}"),
        "payroll": (R.choice(["Salary revision letter", "Update your payroll information", "Bonus payment details"]),
                    f"{g}\n\nHR has published your salary revision. Log in with your corporate credentials to view and acknowledge: {link}.{pressure}"),
    }
    subject, body = T[kind]
    if R.random() < 0.15:
        subject = subject.upper()
    sender_dom = dom
    while sender_dom[0].isdigit() or sender_dom in ("bit.ly", "tinyurl.com", "cutt.ly", "rb.gy", "t.ly"):
        sender_dom = lookalike(brand)
    if kind.startswith("bec"):
        sender = f'"{name()}" <{R.choice(["ceo.office", "md.desk", "director"])}{R.randint(1, 99)}@{R.choice(["gmail.com", "outlook.com", "yahoo.com"])}>'
    else:
        sender = f'"{brand}" <{R.choice(["support", "security", "no-reply", "service", "alert"])}@{sender_dom}>'
    return subject, body + sign(brand) + footer(brand), sender


def main(n_per_class: int = 4000) -> None:
    brands = list(BRANDS)
    rows = []
    for _ in range(n_per_class):
        s, b, f = legit(R.choice(brands))
        rows.append({"subject": s, "body": b, "sender": f, "label": 0})
        s, b, f = phish(R.choice(brands))
        rows.append({"subject": s, "body": b, "sender": f, "label": 1})
    df = pd.DataFrame(rows).sample(frac=1, random_state=RANDOM_SEED)
    out = RAW_DIR / "Modern_Synthetic.csv"
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
