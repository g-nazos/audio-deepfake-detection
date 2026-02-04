raw_text = (
    "Machine learning systems acquire recognition capabilities through processes "
    "that parallel human learning patterns. Object recognition develops through "
    "exposure to numerous examples, while natural language processing systems "
    "acquire linguistic capabilities through extensive textual analysis. These learning "
    "approaches operationalize theories of intelligence developed in AI research, "
    "building on mathematical foundations that we establish systematically throughout this text. "
    "The distinction between AI as research vision and ML as engineering methodology carries significant implications for system design. Modern ML's datadriven approach requires infrastructure capable of collecting, processing, and "
    "learning from data at massive scale. Machine learning emerged as a practical approach to artificial intelligence through extensive research and major "
    "paradigm shifts2, transforming theoretical principles about intelligence into "
    "functioning systems that form the algorithmic foundation of todays intelligent "
    "capabilities.The evolution from rule-based AI to data-driven ML represents one of the "
    "most significant shifts in computing history. This transformation explains why "    
    "ML systems engineering has emerged as a discipline: the path to intelligent "
    "systems now runs through the engineering challenge of building systems that "
    "can effectively learn from data at massive scale. "
    "Before exploring how we arrived at modern machine learning systems, we "
    "must first establish what we mean by an “ML system.” This definition provides "
    "the conceptual framework for understanding both the historical evolution and "
    "contemporary challenges that follow. "
    "No universally accepted definition of machine learning systems exists, reflecting the field’s rapid evolution and multidisciplinary nature. However, building "
    "on our understanding that modern ML relies on data-driven approaches at "
    "massive scale, we can define an ML system as a computational system that "
    "learns from data. A Machine Learning System is an integrated computing system comprising three core components: data that guides algorithmic behavior, "
    "learning algorithms that extract patterns from this data, and computing infrastructure that enables both the learning process (i.e., training) and the application of learned knowledge (i.e., inference/serving). "
    "Together, these components create a computing system capable of making predictions, generating content, or taking actions based on learned patterns."
    "Space exploration provides an apt analogy for these relationships. Algorithm developers resemble astronauts exploring new frontiers and making "
    "discoveries. Data science teams function like mission control specialists ensuring constant flow of critical information and resources for mission operations. "
    "Computing infrastructure engineers resemble rocket engineers designing and "
    "building systems that enable missions. Just as space missions require seamless "
    "integration of astronauts, mission control, and rocket systems, machine learning systems demand careful orchestration of algorithms, data, and computing "
    "infrastructure."
)
print(len(raw_text))

def paragraphs_from_text(text, words_per_paragraph=13):
    words = text.split()
    result = []
    for i in range(0, len(words), words_per_paragraph):
        chunk = words[i : i + words_per_paragraph]
        result.append(" ".join(chunk))
    return result


paragraphs = paragraphs_from_text(raw_text)
print(len(paragraphs))
print(paragraphs)
