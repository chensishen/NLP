from neo4j import GraphDatabase, Driver, RoutingControl

def add_friend(driver: Driver, db: str, name1: str, name2: str):
	driver.execute_query(
		"""
			MERGE (a:Person {name: $name1})
			MERGE (b:Person {name: $name2})
			MERGE (a) -[:FRIEND]-> (b)
			MERGE (b) -[:FRIEND]-> (a)
		""",
		name1=name1, name2=name2, database_=db
	)


def print_friends(driver: Driver, db: str, name: str):
	records, summary, keys = driver.execute_query(
		"""
			MATCH (a:Person)-[:FRIEND]->(friend:Person) WHERE a.name = $name 
			RETURN friend.name as name
			ORDER BY friend.name
		""",
		name=name, database_=db,
		routing_=RoutingControl.READ  # 当前是查询操作，默认该参数为 WRITE
	)
	print("summary",summary)
	print("keys",keys)
	for record in records:
		print(type(record))
		print(record)
		print(record['name'])

def print_friends2(driver: Driver, db: str, name: str):
	records, summary, keys = driver.execute_query(
		"""
			MATCH l = (a:Person)-[:FRIEND]->(friend:Person) WHERE a.name = $name 
			RETURN l
		""",
		name=name, database_=db,
		routing_=RoutingControl.READ  # 当前是查询操作，默认该参数为 WRITE
	)
	print("summary",summary)
	print("keys",keys)
	print(type(records[0]))
	print(records[0])
	print(type(records[0][0]))
	print(records[0][0])

if __name__ == '__main__':
	url = "bolt://127.0.0.1:7687"  # 连接图数据库的url
	db = "neo4j"  # 图数据库的database 名称字符串
	auth = ("neo4j", "12345678")  # 用户名和密码

	# noinspection PyArgumentList
	with GraphDatabase.driver(url, auth=auth) as driver:
		add_friend(driver, db, "小明", "小红")
		add_friend(driver, db, "小明", "小华")
		add_friend(driver, db, "小明", "小沪")
		add_friend(driver, db, "小华", "张三")
		add_friend(driver, db, "张三", "王五")
		# print_friends(driver, db, "小明")
		print_friends2(driver, db, "小明")