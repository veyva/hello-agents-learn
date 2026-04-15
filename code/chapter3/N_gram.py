import collections

corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens = len(tokens)

# P(datawhale)
count_datawhale = tokens.count("datawhale")
p_datawhale = count_datawhale / total_tokens

# P(agent|datawhale)
bigrams = zip(tokens, tokens[1:])
bigram_counts = collections.Counter(bigrams)
count_datawhale_agent = bigram_counts[('datawhale', 'agent')]
p_agent_given_datawhale = count_datawhale_agent / count_datawhale

# P(learn|agent)
count_agent_learn = bigram_counts[('agent', 'learns')]
count_agent = tokens.count("agent")
p_learn_given_agent = count_agent_learn / count_agent 

# output
p_sentence = p_datawhale * p_agent_given_datawhale * p_learn_given_agent