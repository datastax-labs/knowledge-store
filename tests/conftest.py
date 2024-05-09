import secrets
from typing import Iterator

import pytest
from cassandra.cluster import Cluster, Session
from langchain_core.language_models import BaseChatModel
from testcontainers.core.container import DockerContainer
from testcontainers.core.waiting_utils import wait_for_logs

from knowledge_content_graph.embedding import Embedding, InstructorEmbedding
from knowledge_content_graph.knowledge_graph import ContentGraph


@pytest.fixture(scope="session")
def db_keyspace() -> str:
    return "default_keyspace"


@pytest.fixture(scope="session")
def cassandra_port(db_keyspace: str) -> Iterator[int]:
    # TODO: Allow running against local Cassandra and/or Astra using pytest option.
    cassandra = DockerContainer("cassandra:5")
    cassandra.with_exposed_ports(9042)
    cassandra.with_env(
        "JVM_OPTS",
        "-Dcassandra.skip_wait_for_gossip_to_settle=0 -Dcassandra.initial_token=0",
    )
    cassandra.with_env("HEAP_NEWSIZE", "128M")
    cassandra.with_env("MAX_HEAP_SIZE", "1024M")
    cassandra.with_env("CASSANDRA_ENDPOINT_SNITCH", "GossipingPropertyFileSnitch")
    cassandra.with_env("CASSANDRA_DC", "datacenter1")
    cassandra.start()
    wait_for_logs(cassandra, "Startup complete")
    cassandra.get_wrapped_container().exec_run(
        (
            f"""cqlsh -e "CREATE KEYSPACE {db_keyspace} WITH replication = """
            '''{'class': 'SimpleStrategy', 'replication_factor': '1'};"'''
        )
    )
    port = cassandra.get_exposed_port(9042)
    print(f"Cassandra started. Port is {port}")
    yield port
    cassandra.stop()


@pytest.fixture(scope="session")
def db_session(cassandra_port: int) -> Session:
    print(f"Connecting to cassandra on {cassandra_port}")
    cluster = Cluster(
        port=cassandra_port,
    )
    return cluster.connect()


@pytest.fixture(scope="session")
def local_llm() -> BaseChatModel:
    from langchain_community.llms import Ollama
    return Ollama(model="llama2")


@pytest.fixture(scope="session")
def local_embedding() -> Embedding:
    return InstructorEmbedding()


class DataFixture:
    def __init__(self,
                 session: Session,
                 keyspace: str,
                 text_embedding: Embedding) -> None:
        self.session = session
        self.keyspace = "default_keyspace"
        self.uid = secrets.token_hex(8)
        self.content_table = f"content_{self.uid}"
        self.graph = ContentGraph(
            text_embedding,
            content_table=self.content_table,
            session=session,
            keyspace=keyspace,
        )

    def drop(self):
        self.session.execute(f"DROP TABLE IF EXISTS {self.keyspace}.{self.content_table};")

@pytest.fixture()
def fresh_fixture(db_session: Session,
                  db_keyspace: str,
                  local_embedding) -> Iterator[DataFixture]:
    data = DataFixture(session=db_session, keyspace=db_keyspace, text_embedding=local_embedding)
    yield data
    data.drop()