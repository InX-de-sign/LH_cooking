#!/usr/bin/env python
"""Verify the configured Azure OpenAI deployment before or after a model switch.

Run this instead of hand-writing a curl command. It loads chatbot/config.py, so
it sends exactly the parameters the chatbot will send -- including the family
switch between `max_tokens` (gpt-4o) and `max_completion_tokens`
(gpt-5/o-series). A curl command typed by hand tests a request the application
never makes.

    python scripts/check_model.py
    python scripts/check_model.py --model gpt-4o-mini     # try another deployment
    python scripts/check_model.py --budget 50             # reproduce the empty-reply bug

Exits non-zero on failure, so it is usable as a deployment gate.

WHAT IT IS LOOKING FOR
----------------------
The dangerous failure when moving to gpt-5 is silent. The completion budget
pays for hidden reasoning tokens as well as the visible answer, so a budget
sized for a one-sentence reply returns an EMPTY string with
finish_reason="length" -- no error, no exception, just a guide that says
nothing. This script calls that out explicitly rather than printing a happy
"200 OK".
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CHATBOT = os.path.join(os.path.dirname(HERE), "chatbot")
sys.path.insert(0, CHATBOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="deployment name to test (default: from config)")
    parser.add_argument("--budget", type=int,
                        help="override the completion budget, bypassing the floor")
    parser.add_argument("--effort", choices=("minimal", "low", "medium", "high"),
                        help="override reasoning_effort (gpt-5 only)")
    parser.add_argument("--prompt", default="Say hello in one short sentence.")
    args = parser.parse_args()

    if args.model:
        os.environ["AZURE_DEPLOYMENT_NAME"] = args.model

    from config import load_azure_openai_config
    from api_client import base_url, is_hkust_genai

    config = load_azure_openai_config()
    if not config.configured:
        print("FAIL  Fill OPENAI_API_KEY in chatbot/.env, then run this check again.")
        return 2
    kwargs = config.completion_kwargs()

    if args.budget:
        # Deliberately bypass MIN_REASONING_BUDGET. Going through
        # completion_kwargs() would clamp the value back up to the floor, which
        # is right for the application and useless for demonstrating what the
        # floor prevents.
        from config import MIN_REASONING_BUDGET

        key = "max_completion_tokens" if config.is_reasoning_model else "max_tokens"
        kwargs[key] = args.budget
        if config.is_reasoning_model and args.budget < MIN_REASONING_BUDGET:
            print(f"note        --budget {args.budget} is below the"
                  f" MIN_REASONING_BUDGET floor of {MIN_REASONING_BUDGET};"
                  f" sending it raw to show what the floor prevents.")
            if kwargs.get("reasoning_effort") == "minimal" and not args.effort:
                print("            (at effort=minimal a small budget is usually"
                      " still fine -- add --effort medium to see it fail)")
            print()

    if args.effort:
        kwargs["reasoning_effort"] = args.effort

    print(f"endpoint    {base_url(config)}")
    print(f"deployment  {config.deployment_name}")
    if not is_hkust_genai(config):
        print(f"api-version {config.api_version}")
    print(f"reasoning   {config.is_reasoning_model}")
    print(f"parameters  {json.dumps(kwargs)}")
    print()

    if is_hkust_genai(config):
        url = f"{base_url(config)}/chat/completions"
    else:
        url = (f"{config.azure_endpoint}/openai/deployments/{config.deployment_name}"
               f"/chat/completions?api-version={config.api_version}")
    body = {"messages": [{"role": "user", "content": args.prompt}]}
    if is_hkust_genai(config):
        body["model"] = config.deployment_name
    body.update(kwargs)

    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"api-key": config.api_key, "Content-Type": "application/json"},
    )

    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400].replace(config.api_key, "[hidden]")
        print(f"FAIL  HTTP {exc.code} after {time.time() - started:.1f}s")
        print(f"      {detail}")
        if exc.code == 404:
            if is_hkust_genai(config):
                print("\n      Check the HKUST GenAI base URL and model name.")
            else:
                print("\n      Check the Azure deployment and api-version.")
        elif exc.code == 403:
            print("\n      A 403 means the deployment name is not one this key"
                  "\n      can reach -- either it does not exist, or it is outside"
                  "\n      the model groups the message above lists.")
        elif exc.code == 401:
            print("\n      Check OPENAI_API_KEY in chatbot/.env. An exported OS value"
                  "\n      overrides that file. The repository-root .env is not read.")
        elif exc.code == 400 and "temperature" in detail:
            print("\n      gpt-5 accepts temperature=1 only. config.completion_kwargs()"
                  "\n      handles this; a call site passing temperature directly does not.")
        return 1
    except Exception as exc:
        print(f"FAIL  {type(exc).__name__}: {exc}")
        return 1

    elapsed = time.time() - started
    choice = (payload.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    finish = choice.get("finish_reason")
    usage = payload.get("usage") or {}
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0)

    print(f"latency     {elapsed:.2f}s")
    print(f"finish      {finish}")
    print(f"tokens      completion={usage.get('completion_tokens')} "
          f"reasoning={reasoning}")
    print(f"reply       {content!r}")
    print()

    if not content.strip():
        print("FAIL  The model returned an EMPTY reply.")
        if finish == "length":
            budget = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
            print(f"      finish_reason=length with a budget of {budget} tokens and"
                  f" {reasoning} spent on reasoning.")
            print("      The budget has to cover reasoning AND the visible answer.")
            print("      Raise AZURE_MAX_COMPLETION_TOKENS, or keep"
                  " AZURE_REASONING_EFFORT=minimal.")
        return 1

    print(f"OK    {config.deployment_name} is answering correctly.")
    if config.is_reasoning_model and reasoning == 0:
        print("      reasoning_tokens=0 confirms reasoning_effort=minimal is in"
              " effect, which is what the voice path wants.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
