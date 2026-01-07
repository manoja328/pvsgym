from llm_lib import OpenAIBackend

ft_model_id = "ft:gpt-4.1-nano-2025-04-14:chalo:pvs-exp2:Bo3U8j86"
llm = OpenAIBackend(model= ft_model_id, temperature=0.7)

query = "sech_0: LEMMA sech(0) = 1"

response = llm.query(query)
print(response)