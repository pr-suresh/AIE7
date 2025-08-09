#### ❓ Question:

What is the purpose of the `chunk_overlap` parameter when using `RecursiveCharacterTextSplitter` to prepare documents for RAG, and what trade-offs arise as you increase or decrease its value?

##### ✅ Answer:

chunk_overlap repeats the last N characters/tokens of chunk i at the start of chunk i+1. This preserves context that spans chunk boundaries so sentences/facts split across chunks still appear together during retrieval and generation.

When you increase chunk_overlap, it gives better recall for boundary spanning facts but the trade off is higher embedding cost and latency, more duplicate hits can reduce the precision and waste context window with repeated text

When you decrease chunk_overlap, fewer chunks leads to faster embedding, smaller vector store and less duplicate context in prompts but the trade off is that more boundary breaks and lower recall

#### ❓ Question:

Your retriever is configured with `search_kwargs={"k": 5}`. How would adjusting `k` likely affect RAGAS metrics such as Context Precision and Context Recall in practice, and why?

##### ✅ Answer:

Context precision increases when k decreases and it decreases when k increases.
As you add lower-ranked chunks, the share of truly relevant contexts usually drops. When you keep only highest similarity chunks, which tend to be more relevant on average.

Context Recall increases when k increases and it decreases when k decreases. More retrieved chunks means higher chance that all necessary evidence is included. However when k decreases you risk missing parts of the needed evidence if they’re not in the very top results.

Rankings have a relevance tail: lower-ranked results are less likely to be relevant, helping recall but diluting precision.
Larger k can also hurt downstream LLM focus (more noise/duplicates), which may indirectly reduce measured precision or answer quality.



#### ❓ Question:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

##### ✅ Answer:

In agent_helpful, the helpfulness node runs right after the agent node when the latest agent response does not request tool calls. It evaluates the answer’s helpfulness and decides to loop or end.

In agent, there is no evaluator; if the agent response has no tool calls, execution terminates immediately.

When to route back vs. terminate
Route back to agent if the evaluator returns unhelpful (HELPFULNESS:N) → decision "continue".
Terminate if the evaluator returns helpful (HELPFULNESS:Y) → decision "end", or if the loop-limit is hit (HELPFULNESS:END) → END.