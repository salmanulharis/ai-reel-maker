"""Video brief: what a narrated video says and shows, written from the finished analysis.

The analysis already knows the moments (candidates with clean cut points), what is said, what
the picture shows and what can be heard. A brief turns that into a short story: a title, a
hook, a list of beats (each a few chosen moments with a line of narration and some on-screen
text) and a call to action. Two engines can write it, chosen by the user:

- ``local``: Ollama on this computer (the default);
- ``claude``: the Claude Code CLI on the user's Claude subscription.

Both read the same evidence (app.brief.evidence), reply in the same format (app.brief.schema),
and every reply is checked and repaired by code (app.brief.validate), so a brief always fits the
length and only uses moments that exist. Moments are chosen by their alias only, never by time.
"""
