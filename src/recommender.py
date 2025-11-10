# src/recommender.py
from typing import Any, Dict, List, Optional, Callable

# Try to import v1 PromptTemplate; if not available, we'll fallback to a simple wrapper
try:
    from langchain_core.prompts.prompt import PromptTemplate  # v1-style
except Exception:
    PromptTemplate = None

# Helper: simple fallback PromptTemplate-like object
class _SimplePromptTemplate:
    def __init__(self, template: str, input_variables: List[str] = None):
        self.template = template
        self.input_variables = input_variables or []

    def format(self, **kwargs) -> str:
        # safe format: replace known keys in curly braces
        txt = self.template
        for k, v in kwargs.items():
            txt = txt.replace("{" + k + "}", str(v))
        return txt

# Robust LLM caller: attempts common invocation patterns and returns a string
# Replace your existing _call_llm function with this one in src/recommender.py

def _call_llm(llm: Any, prompt_text: str) -> str:
    """
    Robust invocation helper that supports:
      - LangChain Runnable chat models like langchain_groq.ChatGroq (invoke with messages)
      - LLMs exposing .generate, .predict, callable, .invoke({...}) forms
    Returns a plain string.
    """
    # 0) Quick helper to extract text from various return shapes
    def _extract_text_from_result(res: Any) -> str:
        # direct string
        if isinstance(res, str):
            return res
        # dict-like
        if isinstance(res, dict):
            for k in ("output", "result", "text", "content"):
                if k in res:
                    return str(res[k])
            # some runnables return {'generations': ...}
            if "generations" in res:
                try:
                    gen = res["generations"]
                    return str(gen[0][0].text)
                except Exception:
                    return str(res)
            return str(res)
        # objects with properties
        if hasattr(res, "generations"):
            try:
                return str(res.generations[0][0].text)
            except Exception:
                pass
        if hasattr(res, "text"):
            return str(res.text)
        if hasattr(res, "content"):
            return str(res.content)
        # fallback
        return str(res)

    # 1) If it's the ChatGroq runnable (or any chat runnable), prefer invoke with message list
    try:
        cls_name = getattr(llm, "__class__", None).__name__
        module_name = getattr(llm, "__class__", None).__module__ or ""
        is_chatgroq = ("ChatGroq" in str(cls_name)) or ("langchain_groq" in module_name)
        # also treat generically as chat runnable if .invoke exists and llm expects messages
        if is_chatgroq or (hasattr(llm, "invoke") and "chat" in (module_name.lower() or "") ):
            # prepare message list: you can expand to include a system prompt if you want
            messages = [{"role": "user", "content": prompt_text}]
            try:
                res = llm.invoke(messages)
                return _extract_text_from_result(res)
            except TypeError:
                # some runnables expect a dict or different kw; try invoke with a dict wrapper
                try:
                    res = llm.invoke({"messages": messages})
                    return _extract_text_from_result(res)
                except Exception:
                    pass
            except Exception:
                # fall through to other attempts
                pass
    except Exception:
        # ignore detection errors and try other patterns
        pass

    # 2) try callable style (some LLM wrappers implement __call__)
    try:
        if callable(llm):
            out = llm(prompt_text)
            txt = _extract_text_from_result(out)
            if txt:
                return txt
    except Exception:
        pass

    # 3) try .predict(prompt) pattern
    try:
        if hasattr(llm, "predict"):
            out = llm.predict(prompt_text)
            return _extract_text_from_result(out)
    except Exception:
        pass

    # 4) try .generate([prompt]) pattern (some LangChain LLMs)
    try:
        if hasattr(llm, "generate"):
            out = llm.generate([prompt_text])
            return _extract_text_from_result(out)
    except Exception:
        pass

    # 5) try generic .invoke with a simple input dict (some runnables accept {"input": "..."})
    try:
        if hasattr(llm, "invoke"):
            out = llm.invoke({"input": prompt_text})
            return _extract_text_from_result(out)
    except Exception:
        pass

    # 6) try .invoke with {"messages": [...]} (generic chat shape)
    try:
        if hasattr(llm, "invoke"):
            msgs = [{"role": "user", "content": prompt_text}]
            out = llm.invoke({"messages": msgs})
            return _extract_text_from_result(out)
    except Exception:
        pass

    # If we reach here, nothing worked
    raise RuntimeError(
        "Unable to invoke the LLM: unsupported interface. "
        f"LLM object: {getattr(llm, '__class__', llm)}. "
        "Tried invoke(messages), invoke({'messages':...}), callable(...), predict(...), generate(...)."
    )

    # 1) If LLM is callable (most LangChain LLM wrappers are), try direct call
    try:
        if callable(llm):
            out = llm(prompt_text)
            # if returns a dict-like or object, try to extract text
            if isinstance(out, str):
                return out
            try:
                # langchain LLMs sometimes return a dict-like or object with "text" or "content"
                if isinstance(out, dict):
                    for key in ("text", "content", "result", "output"):
                        if key in out:
                            return str(out[key])
                # try attribute access
                if hasattr(out, "text"):
                    return str(out.text)
            except Exception:
                pass
            # fallback to stringification
            return str(out)
    except TypeError:
        # not callable in that way — fallthrough
        pass
    except Exception:
        # fallthrough to other attempts
        pass

    # 2) try .predict
    try:
        if hasattr(llm, "predict"):
            ret = llm.predict(prompt_text)
            if isinstance(ret, str):
                return ret
            return str(ret)
    except Exception:
        pass

    # 3) try .generate([...]) pattern used in some LangChain LLMs
    try:
        if hasattr(llm, "generate"):
            # many .generate return a GenerationResult with .generations[0][0].text
            gen = llm.generate([prompt_text])
            # guard for multiple possible shapes
            if hasattr(gen, "generations"):
                try:
                    return str(gen.generations[0][0].text)
                except Exception:
                    return str(gen)
            return str(gen)
    except Exception:
        pass

    # 4) try .invoke
    try:
        if hasattr(llm, "invoke"):
            ret = llm.invoke({"input": prompt_text})
            # get string from ret
            if isinstance(ret, str):
                return ret
            if isinstance(ret, dict):
                for key in ("output", "result", "text"):
                    if key in ret:
                        return str(ret[key])
            return str(ret)
    except Exception:
        pass

    # If nothing worked, raise a helpful error
    raise RuntimeError("Unable to invoke the LLM: unsupported interface. LLM object: {}".format(type(llm)))


class Recommender:
    def __init__(self, retriever: Any, llm: Any, prompt: Optional[Any] = None, top_k: int = 5):
        """
        retriever: object returned by vectorstore.as_retriever() or similar
        llm: an instantiated LLM object (ChatGroq or any LLM wrapper). The function will try several call patterns.
        prompt: PromptTemplate (langchain_core) OR a simple template string OR None.
                If None, a generic prompt template will be used.
        top_k: number of documents to fetch from retriever
        """
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k

        # normalize prompt into something with .format(**kwargs)
        if prompt is None:
            default_tmpl = (
                "You are a TV show recommendation assistant.\n\n"
                "Context: {context}\n\n"
                "Documents:\n{documents}\n\n"
                "Question: {question}\n\n"
                "Provide a short recommendation answer (with reasons) and optionally list source titles."
            )
            self.prompt = _SimplePromptTemplate(default_tmpl, ["context", "documents", "question"])
        else:
            # Accept either a PromptTemplate-like object (with .format) or a plain string
            if PromptTemplate is not None and isinstance(prompt, PromptTemplate):
                self.prompt = prompt
            elif hasattr(prompt, "format") and callable(getattr(prompt, "format")):
                # It already behaves like a PromptTemplate
                self.prompt = prompt
            elif isinstance(prompt, str):
                self.prompt = _SimplePromptTemplate(prompt, ["context", "documents", "question"])
            else:
                # as a last resort, stringify it
                self.prompt = _SimplePromptTemplate(str(prompt), ["context", "documents", "question"])

    def _get_documents_text(self, query: str) -> str:
        # Try common retriever methods in order
        docs = None
        for method_name in ("get_relevant_documents", "get_relevant_documents", "retrieve", "get_documents", "get_docs"):
            try:
                method = getattr(self.retriever, method_name, None)
                if method:
                    docs = method(query)
                    break
            except Exception:
                docs = None

        if docs is None:
            # Try calling the retriever directly (some runnables are callable)
            try:
                docs = self.retriever(query)
            except Exception:
                docs = None

        if not docs:
            return ""

        # docs may be a list of Document objects or strings
        # extract text (common attribute is 'page_content' or 'content' or just the item)
        pieces: List[str] = []
        for i, d in enumerate(docs[: self.top_k]):
            text = ""
            if isinstance(d, str):
                text = d
            else:
                # common attributes
                for attr in ("page_content", "content", "text"):
                    if hasattr(d, attr):
                        try:
                            text = getattr(d, attr)
                            break
                        except Exception:
                            pass
                # fallback to string conversion
                if not text:
                    try:
                        text = str(d)
                    except Exception:
                        text = ""
            # include a small header if available (source, metadata.title)
            header = ""
            try:
                if hasattr(d, "metadata") and isinstance(d.metadata, dict):
                    title = d.metadata.get("title") or d.metadata.get("source") or d.metadata.get("name")
                    if title:
                        header = f"Source: {title}\n"
            except Exception:
                header = ""
            if header:
                pieces.append(header + text)
            else:
                pieces.append(text)
        return "\n\n---\n\n".join(pieces)

    def get_recommendation(self, question: str, context: str = "") -> str:
        # 1) collect top-k docs text
        docs_text = self._get_documents_text(question)

        # 2) build prompt text
        prompt_text = self.prompt.format(context=context, documents=docs_text, question=question)

        # 3) call the LLM and return text
        answer = _call_llm(self.llm, prompt_text)
        return answer
