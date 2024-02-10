from langchain.chains import VectorDBQA, RetrievalQA, ConversationalRetrievalChain


class CustomVectorDBQA(VectorDBQA):
    def invoke(self, input_data):
        result = super().invoke(input_data)
        # for res in result:
        #     print(result.get("source_documents"))
        #     print(res)
        #     break
        if not result.get("source_documents"):
            return {"answer": "Answer not available"}

        return result


class CustomRetrivalQA(RetrievalQA):
    def invoke(self, input_data):
        result = super().invoke(input_data)
        print(f"Input data :- {input_data}")
        print(f'Source :- {result.get("source_documents")}')
        # for res in result:
        #         # print(True)
        #     print(result.get("source_documents"))
        #     print(res)
        #     break
        if not result.get("source_documents"):
            return {"result": "I don't know the answer"}

        return result


class CustomConversationalRetrievalChain(ConversationalRetrievalChain):
    def invoke(self, input_data):
        result = super().invoke(input_data)
        # for res in result:
        #     print(result.get("source_documents"))
        #     print(res)
        #     break
        # print(f"Input data :- {input_data}")
        # print(f'Source :- {result.get("source_documents")}')

        if not result.get("source_documents"):
            return {"answer": "I don't know the answer"}

        return result
