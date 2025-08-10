from typing import Annotated
from fastmcp import FastMCP
from fastmcp.server.auth.providers.bearer import BearerAuthProvider, RSAKeyPair
import markdownify
from mcp import ErrorData, McpError
from mcp.server.auth.provider import AccessToken
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, TextContent
from openai import BaseModel
from pydantic import AnyUrl, Field
import readabilipy
from pathlib import Path
import os
import PyPDF2
import docx  # This library is called python-docx when you install it

# --- Your Personal Information ---
# This is correct with your provided key.
TOKEN = "6cb39e43b15d"
# Make sure this is your correct phone number in the format {country_code}{number}
MY_NUMBER = "917357168831"
# --- Path to your resume file ---
# IMPORTANT: Use a raw string (r"...") for Windows paths to avoid errors.
# Make sure this file actually exists at this location on your computer.
RESUME_FILE_PATH = "PrachiRathiResume.pdf"


# --- Helper Functions to read resume files ---
def convert_pdf_to_text(file_path: str) -> str:
    """Reads a PDF file and returns its text content."""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() or ""  # Add 'or ""' to handle empty pages
            return text
    except Exception as e:
        # Return the error message to be displayed by the tool
        return f"Error reading PDF file: {e}"

def convert_docx_to_text(file_path: str) -> str:
    """Reads a .docx file and returns its text content."""
    try:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"Error reading DOCX file: {e}"


class RichToolDescription(BaseModel):
    description: str
    use_when: str
    side_effects: str | None


class SimpleBearerAuthProvider(BearerAuthProvider):
    def __init__(self, token: str):
        k = RSAKeyPair.generate()
        super().__init__(public_key=k.public_key, jwks_uri=None, issuer=None, audience=None)
        self.token = token

    async def load_access_token(self, token: str) -> AccessToken | None:
        if token == self.token:
            return AccessToken(token=token, client_id="unknown", scopes=[], expires_at=None)
        return None


class Fetch:
    IGNORE_ROBOTS_TXT = True
    USER_AGENT = "Puch/1.0 (Autonomous)"

    @classmethod
    async def fetch_url(cls, url: str, user_agent: str, force_raw: bool = False) -> tuple[str, str]:
        from httpx import AsyncClient, HTTPError
        async with AsyncClient() as client:
            try:
                response = await client.get(url, follow_redirects=True, headers={"User-Agent": user_agent}, timeout=30)
            except HTTPError as e:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url}: {e!r}"))
            if response.status_code >= 400:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message=f"Failed to fetch {url} - status code {response.status_code}"))
            page_raw = response.text
        content_type = response.headers.get("content-type", "")
        is_page_html = ("<html" in page_raw[:100] or "text/html" in content_type or not content_type)
        if is_page_html and not force_raw:
            return cls.extract_content_from_html(page_raw), ""
        return (page_raw, f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n")

    @staticmethod
    def extract_content_from_html(html: str) -> str:
        ret = readabilipy.simple_json.simple_json_from_html_string(html, use_readability=True)
        if not ret["content"]:
            return "<error>Page failed to be simplified from HTML</error>"
        return markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)


mcp = FastMCP(
    "My MCP Server",
    auth=SimpleBearerAuthProvider(TOKEN),
)

ResumeToolDescription = RichToolDescription(
    description="Serve your resume in plain markdown.",
    use_when="Puch (or anyone) asks for your resume; this must return raw markdown, no extra formatting.",
    side_effects=None,
)


@mcp.tool(description=ResumeToolDescription.model_dump_json())
async def resume() -> str:
    """
    Finds your resume file, converts it to markdown text, and returns it.
    This function now handles PDF, DOCX, and TXT files automatically.
    """
    try:
        # Check if the file exists before proceeding
        if not os.path.exists(RESUME_FILE_PATH):
            return "Error: Resume file not found. Please check the RESUME_FILE_PATH variable in the code."

        raw_text = ""
        # Process the file based on its extension
        file_ext = RESUME_FILE_PATH.lower().split('.')[-1]

        if file_ext == 'pdf':
            raw_text = convert_pdf_to_text(RESUME_FILE_PATH)
        elif file_ext == 'docx':
            raw_text = convert_docx_to_text(RESUME_FILE_PATH)
        elif file_ext in ['txt', 'md']:
            with open(RESUME_FILE_PATH, "r", encoding="utf-8") as f:
                raw_text = f.read()
        else:
            return "Error: Unsupported resume file format. Please use .pdf, .docx, .txt, or .md."

        # Convert the extracted text to markdown format for the LLM
        if "Error:" in raw_text: # Pass along errors from helper functions
            return raw_text
        
        markdown_resume = markdownify.markdownify(raw_text, heading_style=markdownify.ATX)
        return markdown_resume

    except Exception as e:
        return f"An unexpected error occurred in the resume tool: {e}"


@mcp.tool
async def validate() -> str:
    """
    NOTE: This tool must be present in an MCP server used by puch.
    """
    return MY_NUMBER


FetchToolDescription = RichToolDescription(
    description="Fetch a URL and return its content.",
    use_when="Use this tool when the user provides a URL and asks for its content, or when the user wants to fetch a webpage.",
    side_effects="The user will receive the content of the requested URL in a simplified format, or raw HTML if requested.",
)


@mcp.tool(description=FetchToolDescription.model_dump_json())
async def fetch(
    url: Annotated[AnyUrl, Field(description="URL to fetch")],
    max_length: Annotated[int, Field(default=5000, description="Maximum number of characters to return.", gt=0, lt=1000000)] = 5000,
    start_index: Annotated[int, Field(default=0, description="On return output starting at this character index, useful if a previous fetch was truncated and more context is required.", ge=0)] = 0,
    raw: Annotated[bool, Field(default=False, description="Get the actual HTML content if the requested page, without simplification.")] = False,
) -> list[TextContent]:
    """Fetch a URL and return its content."""
    url_str = str(url).strip()
    if not url:
        raise McpError(ErrorData(code=INVALID_PARAMS, message="URL is required"))

    content, prefix = await Fetch.fetch_url(url_str, Fetch.USER_AGENT, force_raw=raw)
    original_length = len(content)
    if start_index >= original_length:
        content = "<error>No more content available.</error>"
    else:
        truncated_content = content[start_index : start_index + max_length]
        if not truncated_content:
            content = "<error>No more content available.</error>"
        else:
            content = truncated_content
            actual_content_length = len(truncated_content)
            remaining_content = original_length - (start_index + actual_content_length)
            if actual_content_length == max_length and remaining_content > 0:
                next_start = start_index + actual_content_length
                content += f"\n\n<error>Content truncated. Call the fetch tool with a start_index of {next_start} to get more content.</error>"
    return [TextContent(type="text", text=f"{prefix}Contents of {url}:\n{content}")]


async def main():
    print("Starting MCP server...")
    print(f"Resume tool will attempt to load file from: {RESUME_FILE_PATH}")
    await mcp.run_async(
        "streamable-http",
        host="0.0.0.0",
        port=8085,
    )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())