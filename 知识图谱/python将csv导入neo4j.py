import pandas as pd

if __name__ == '__main__':
    from py2neo import Graph
    from py2neo.cypher import Cursor

    profile = "neo4j://127.0.0.1:7687"  # 连接图数据库的url
    name = "neo4j"  # 图数据库的database 名称字符串


    auth = ("neo4j", "12345678")  # 用户名和密码
    graph = Graph(profile=profile, name=name, auth=auth)

    df = pd.read_csv("./artists3.csv", sep=",", header=None)

    cql_str = """
        MERGE (a:Artist {id:toInteger($id)})
        ON CREATE SET
            a.id = toInteger($id),
            a.name = $name,
            a.year = toInteger($year),
            a.age = toInteger($age),
            a.created = timestamp()
        ON MATCH SET
            a.year = toInteger($year),
            a.age = toInteger($age)
        RETURN a;
    """

    for line in df.values:
        r1 = graph.run(cql_str, id=line[0], name=line[1], year=line[2], age=line[3])
        print(r1.data())