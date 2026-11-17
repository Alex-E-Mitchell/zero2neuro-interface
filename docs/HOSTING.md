# Hosting the class demo on Render

Merge these files into main, then sign into https://dashboard.render.com.
Choose New > Blueprint, connect Alex-E-Mitchell/zero2neuro-interface, and select
main. Render reads render.yaml, selects a Free Python web service, and generates
FLASK_SECRET_KEY. Review the configuration and deploy. Record the actual public
URL shown after the deployment succeeds; do not assume a particular hostname.

If repository access is unavailable, ask a repository administrator to authorize
the Render GitHub integration, or connect the public repository through Render's
Web Service setup. For manual setup use:

- Build: pip install -r requirements-hosting.txt
- Start: gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --access-logfile - --error-logfile -
- Plan: Free
- PYTHON_VERSION: 3.13.11
- FLASK_SECRET_KEY: generate a random secret in Render; never commit it.

The wsgi entry point requires the secret, disables debug mode, and sets secure
session cookies for HTTPS. Local development continues to use app.py on localhost.

## Verify the public deployment

1. Open the public HTTPS URL and visit Network, Data, Experiment, and Review.
2. Generate valid configurations on each page. Confirm Review contains all three.
3. Download experiment.txt and the complete ZIP; compare contents with previews.
4. Submit invalid values and check the validation feedback.
5. Open a private browser window and confirm it cannot see the first session's
   saved configurations.
6. Record the public URL and screenshots for the ticket after these checks pass.

## Demo limitations

Render Free services sleep after 15 minutes without inbound traffic and may take
about a minute to wake. Open the site before a presentation.

The existing config_store keeps configurations in memory. Restarts, redeployments,
and idle shutdowns erase them. Keep one worker and one service instance; regenerate
configurations after a restart. This is a temporary class demo, not durable storage
or a deployment intended for sustained public traffic. A shared persistent store
with expiry is needed before increasing worker count or scaling the service.

The optional upstream parser check requires NEURO_REPOSITORY_PATH to point to a
Zero2Neuro checkout on the server. This basic deployment does not install that
separate framework, so parser validation reports unavailable; the interface's own
form validation and configuration exports still work. Local laptop paths cannot
be used for that variable on Render. Model training is outside this hosting setup.

References: https://render.com/docs/deploy-flask and https://render.com/docs/free
