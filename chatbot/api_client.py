"""Clients for HKUST GenAI and direct Azure OpenAI resources.

HKUST's current GenAI API is OpenAI-compatible at /hkust-genai/v1 and uses an
api-key header. Its older /openai/deployments route is a separate, deprecated
subscription and can reject a valid GenAI key with HTTP 401.
"""

from urllib.parse import urlparse

from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI


def is_hkust_genai(config) -> bool:
    return urlparse(config.azure_endpoint).hostname == "hkust.azure-api.net"


def base_url(config) -> str:
    endpoint = config.azure_endpoint.rstrip("/")
    if is_hkust_genai(config) and urlparse(endpoint).path in ("", "/"):
        return endpoint + "/hkust-genai/v1"
    return endpoint


def create_async_client(config, *, timeout, max_retries):
    if is_hkust_genai(config):
        return AsyncOpenAI(
            api_key=config.api_key,
            base_url=base_url(config),
            default_headers={"api-key": config.api_key},
            timeout=timeout,
            max_retries=max_retries,
        )
    return AsyncAzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.azure_endpoint,
        timeout=timeout,
        max_retries=max_retries,
    )


def create_client(config, *, timeout, max_retries):
    if is_hkust_genai(config):
        return OpenAI(
            api_key=config.api_key,
            base_url=base_url(config),
            default_headers={"api-key": config.api_key},
            timeout=timeout,
            max_retries=max_retries,
        )
    return AzureOpenAI(
        api_key=config.api_key,
        api_version=config.api_version,
        azure_endpoint=config.azure_endpoint,
        timeout=timeout,
        max_retries=max_retries,
    )
