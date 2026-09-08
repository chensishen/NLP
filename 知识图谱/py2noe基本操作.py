
def tt_01(graph):

	# 插入数据
	r1: Cursor = graph.run(
		"""
			CREATE (b:Book {name:$name, price:$book_price}) RETURN b
		""",
		name='深度学习基础',
		book_price=13.78
	)
	print(type(r1))
	print(r1)
	print(r1.data())

def tt_02(graph):
	r2 = graph.run("MATCH (b1:Book {name:$name1}) RETURN b1.name AS name, b1.price AS price", name1='深度学习基础')
	print(r2.to_data_frame())
	print(r2.to_ndarray())

def tt_03(graph):
	r3: Cursor = graph.run("""
	    MATCH
	        (p1:Person {name:'成龙'})-[:参演]->(moive)<-[:参演]-(p2:Person)
	    WITH p2
	    MATCH
	        l = (p2)-[:参演]->(moive2)
	    RETURN l;
	""")

	current_record = r3
	print(type(current_record))
	print(current_record)
	# print(current_record.data())
	# print(current_record.to_data_frame())
	print(current_record.to_ndarray())


def tt_04(graph):
	r3: Cursor = graph.run("""
	    MATCH
	        (p1:Person {name:'成龙'})-[:参演]->(moive)<-[:参演]-(p2:Person)
	    WITH p2
	    MATCH
	        l = (p2)-[:参演]->(moive2)
	    RETURN l;
	""")
	while r3.forward():
		print("=" * 100)
		current_record = r3.current  # 当前这一条记录
		print(type(current_record))
		print(current_record)
		print(current_record[0])



def tt_05(graph):
	r3: Cursor = graph.run("""
	    MATCH
	        (p1:Person {name:'成龙'})-[:参演]->(moive)<-[:参演]-(p2:Person)
	    WITH p2
	    MATCH
	        (p2)-[:参演]->(moive2)
	    RETURN p2,moive2;
	""")
	while r3.forward():
		print("=" * 100)
		current_record = r3.current  # 当前这一条记录
		print(type(current_record))
		print(current_record)
		print(current_record['p2'])
		print(current_record['moive2'])
		print(current_record['moive2']['name'])  # 获取name属性
		print(current_record['moive2'].get('name'))  # 获取name属性
		print(current_record['p2'].get('occupation'))  # 获取occupation属性

if __name__ == '__main__':
	from py2neo import Graph
	from py2neo.cypher import Cursor

	profile = "bolt://127.0.0.1:7687"  # 连接图数据库的url
	name = "neo4j"  # 图数据库的database 名称字符串
	auth = ("neo4j", "12345678")  # 用户名和密码
	graph = Graph(profile=profile, name=name, auth=auth)

	# tt_01(graph)
	# tt_02(graph)
	tt_03(graph)
	pass