import xml.etree.ElementTree as ET
from typing import List
from langchain_core.runnables import RunnableConfig
from langgraph.types import StreamWriter
import logging
from langchain_core.utils.json import parse_json_markdown
from aiq_aira.schema import GeneratedQuery
from aiq_aira.utils import format_citation, log_both, _escape_markdown
from langchain_core.messages import HumanMessage
import html
import json
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


async def check_relevancy(llm: ChatOpenAI, query: str, answer: str, writer: StreamWriter):
    """
    Checks if an answer is relevant to the query using the 'relevancy_checker' prompt, returning JSON
    like { "score": "yes" } or { "score": "no" }.
    """
    logger.info("CHECK RELEVANCY")    
    writer({"relevancy_checker": "\n Starting relevancy check \n"})
    processed_answer_for_display = html.escape(_escape_markdown(answer))

    try:
        async with asyncio.timeout(ASYNC_TIMEOUT):
            response = await llm.ainvoke(
                relevancy_checker.format(document=answer, query=query)
            )
            score = parse_json_markdown(response.content)
            writer({"relevancy_checker": f""" =
    ---
    Relevancy score: {score.get("score")}  
    Query: {query}
    Answer: {processed_answer_for_display}
    """})

            return score
    
    except asyncio.TimeoutError as e:
             writer({"relevancy_checker": f""" 
----------                
LLM time out evaluating relevancy. Query: {query} \n \n Answer: {processed_answer_for_display} 
----------
"""})   
    except Exception as e:
        writer({"relevancy_checker": f"""
---------
Error checking relevancy. Query: {query} \n \n Answer: {processed_answer_for_display} 
---------
"""})
        logger.debug(f"Error parsing relevancy JSON: {e}")

    # default if fails
    return {"score": "yes"}


async def fetch_query_results(
    rag_url: str,
    prompt: str,
    writer: StreamWriter,
    collection: str
):
    """
    Calls the search_rag tool in parallel for each prompt in parallel.
    Returns a list of tuples (answer, citations).
    """
    async with aiohttp.ClientSession() as session:
        result =  await search_rag(session, rag_url, prompt, writer, collection)
        return result



def deduplicate_and_format_sources(
    sources: List[str],
    generated_answers: List[str],
    queries: List[GeneratedQuery]
):
    """
    Convert search agent results into an XML structure <sources><source>...</source></sources>.
    Each <source> has <query> and <answer>.
    """
    logger.info("FORMATTING RESULTS")
    root = ET.Element("sources")

    for q_json, src, gen_ans in zip(
        queries, sources, generated_answers
    ):
        source_elem = ET.SubElement(root, "source")
        query_elem = ET.SubElement(source_elem, "query")
        query_elem.text = str(q_json.query)
        answer_elem = ET.SubElement(source_elem, "answer")
        section_elem = ET.SubElement(source_elem, "section")
        section_elem.text = str(q_json.report_section)
        answer_elem.text = str(gen_ans)
        citation_elem = ET.SubElement(source_elem, "citation")
        citation_elem.text = str(src)

        
    return ET.tostring(root, encoding="unicode")


async def process_single_query(
        query: str,
        config: RunnableConfig,
        writer: StreamWriter,
        collection,
):
    """
    Uses an agent to call tools for a single query.
    The agent returns a tuple of (answers, citations)
    Where answers and citations are concatenated strings of answers and citations
    """

    search_agent = config["configurable"].get("search_agent")
    log_both(f"Agent searching for: {query}", writer, "search_agent")
    add_collection_to_prompt = f"{query} \n If the tool call requires a collection, use the following collection: {collection}"
    response = await search_agent.ainvoke({"input_message": add_collection_to_prompt})
    
    
    try:
        parsed_response = parse_json_markdown(response)
    except Exception as e:
        log_both(f"Error parsing agent response for {query}: {e}", writer, "search_agent")
        return ("No answer found", format_citation(query=query, answer="No answer found", citations="No answer found"))

    answer = parsed_response.get("answer")
    citations = parsed_response.get("citation")
    formatted_citation = format_citation(
        query=query,
        answer=answer,
        citations="".join([
            f" \n \n *{citation.get("tool_name")}* \n ``` \n {citation.get("tool_response")} \n  *Origin*: {citation.get("url")} \n ``` \n \n"
            for citation in citations
        ])
    )

    log_both(formatted_citation, writer, "search_agent")

    return (answer, formatted_citation)