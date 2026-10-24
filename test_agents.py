from src.agents import RetrievalAgent, EvidenceChunk, NewsClassifierAgent

def main():
    r=RetrievalAgent()
    r.add_chunks([EvidenceChunk('sample-source','The city council announced a public transit funding plan on Tuesday.',0),
                  EvidenceChunk('other-source','A local sports team won its final match.',0)])
    results=r.retrieve('What did the city council announce about transit?',2)
    assert results and 'transit' in results[0]['text'].lower(), results
    assert NewsClassifierAgent('models/not-a-real-model').ready is False
    print('Agent smoke test passed: retrieval works and missing model is handled honestly.')
if __name__=='__main__': main()
