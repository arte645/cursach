from langchain.vectorstores import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models.gigachat import GigaChat
from langchain.chains import create_retrieval_chain


class LlmRag:
    def __init__(self, api_key: str, docs_directory: str) -> None:
        self.__load_docs(docs_directory)
        self.__init_llm(api_key)
        self.__set_retrieval_chain()

    def invoke(self, message: str) -> str:
        return self.retrieval_chain.invoke({'input': message})['answer']

    def __load_docs(self, directory):
        loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        self.documents = text_splitter.split_documents(documents)

        embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.db = Chroma.from_documents(self.documents, embedding_function)

    def __init_llm(self, api_key: str) -> None:
        self.llm = GigaChat(
            credentials=api_key,
            model="GigaChat",
            verify_ssl_certs=False,
            profanity_check=False,
        )

    def __set_retrieval_chain(self,):
        retriever = self.db.as_retriever()
        prompt = ChatPromptTemplate.from_template(''' Ты - психолог помогающий бороться с выгоранием \
            Ответь на вопрос пользователя. \
            Используй при этом информацию из контекста. Если в контексте нет \
            информации для ответа, попроси пользователя рассказать про свои проблемы.
            Контекст: {context}
            Вопрос: {input}
            Ответ:'''
        )
        document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt,
                )
        self.retrieval_chain = create_retrieval_chain(retriever, document_chain)
