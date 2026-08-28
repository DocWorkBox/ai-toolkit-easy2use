import re


H3_CAPTION_SECTIONS = (
    "integrated_multimodal_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


def _find_heading(text: str, heading: str, start: int = 0):
    return re.compile(
        rf"^[ \t]*{re.escape(heading)}[ \t]*",
        flags=re.IGNORECASE | re.MULTILINE,
    ).search(text, start)


def extract_first_h3_caption(text: str) -> str:
    """Keep the first complete H3 caption and drop generated chat followups."""
    text = text.strip()
    matches = []
    cursor = 0
    for heading in H3_CAPTION_SECTIONS:
        match = _find_heading(text, heading, cursor)
        if match is None:
            return text
        matches.append(match)
        cursor = match.end()

    music_start = matches[-1].end()
    boundaries = []
    for heading in H3_CAPTION_SECTIONS:
        repeated = _find_heading(text, heading, music_start)
        if repeated is not None:
            boundaries.append(repeated.start())

    role_boundary = re.compile(
        r"^[ \t]*(?:assistant|human|user|system)(?:[ \t]*:.*)?[ \t]*$",
        flags=re.IGNORECASE | re.MULTILINE,
    ).search(text, music_start)
    if role_boundary is not None:
        boundaries.append(role_boundary.start())

    special_token = re.compile(r"<\|(?:im_start|im_end)\|>", re.IGNORECASE).search(
        text, music_start
    )
    if special_token is not None:
        boundaries.append(special_token.start())

    caption_end = min(boundaries, default=len(text))
    contents = (
        text[matches[0].end() : matches[1].start()].strip(),
        text[matches[1].end() : matches[2].start()].strip(),
        text[matches[2].end() : caption_end].strip(),
    )
    return "\n".join(
        f"{heading}\n{content}" if content else heading
        for heading, content in zip(H3_CAPTION_SECTIONS, contents)
    )
