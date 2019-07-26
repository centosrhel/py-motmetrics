from lxml import etree, objectify

# write
E = objectify.ElementMaker(annotate=False)
anno_tree = E.annotation(
    E.folder('VOC2012_instance'),
    E.filename("test.jpg"),
    E.source(
        E.database('COCO'),
        E.annotation('COCO'),
        E.image('COCO'),
        E.url("http://test.jpg")
    ),
    E.size(
        E.width(800),
        E.height(600),
        E.depth(3)
    ),
    E.segmented(0)
)
etree.ElementTree(anno_tree).write("test.xml", pretty_print=True)
# 如果需要在anno_tree的基础上加其他标签的话用append即可
E2 = objectify.ElementMaker(annotate=False)
anno_tree2 = E2.object(
    E.name("person"),
    E.bndbox(
        E.xmin(100),
        E.ymin(200),
        E.xmax(300),
        E.ymax(400)
    ),
    E.difficult(0)
)
anno_tree.append(anno_tree2)
etree.ElementTree(anno_tree).write('test.xml',pretty_print=True)
# one more
E3 = objectify.ElementMaker(annotate=False)
anno_tree3 = E3.object(
    E.name("person"),
    E.bndbox(
        E.xmin(500),
        E.ymin(600),
        E.xmax(700),
        E.ymax(800)
    ),
    E.difficult(0)
)
anno_tree.append(anno_tree3)
etree.ElementTree(anno_tree).write('test.xml',pretty_print=True)

# read
tree = etree.parse('test.xml')
"""
bookstore ~ 选取bookstore元素的所有子节点
bookstore/book ~ 选取bookstore的子节点中的所有book元素
/bookstore ~ 到元素bookstore的绝对路径，即选取元素bookstore
//book ~ 选取所有book元素，不介意它们在文档中的位置
bookstore//book ~ 选取bookstore的子节点及循环子节点中的所有book元素
//@lang ~ 选取名为lang的所有属性
"""
for bbox in tree.xpath('//bndbox'):
    for boundary in bbox.getchildren():
        print(boundary.text) # string类型
# tree.xpath('object'), tree.xpath('/annotation')


###### 读取./lxml.xml ######
xml = etree.parse('lxml.xml')
root = xml.getroot() # 获取根节点
print(root.items()) # 获取全部属性和属性值
print(root.keys()) # 获取全部属性
print(root.get('version','')) # 获取具体某个属性的值

# 遍历所有子节点。
# 若xml文档比较大，还可以使用iterchildren方法。该方法得到一个生成器。
# 可以用dir(root)查得节点对象有什么方法
for node in root.getchildren():
    print(node.tag) # 输出子节点的标签名

# id为3t的target元素中有两段文本，以及文本中间还有个bpt元素
# 获取id属性为3t的target元素，注意后面的[0]
target = root.xpath('//target[@id="3t"]')[0]
print(target.text) #输出该元素的text属性值
"""
将得到"CC"，后面的节点和"cc"获取不到。则text属性是获取到该节点下的第1段文本。
若该节点先是一个节点，再是文字: <target id="3t"><bpt id="3t1"/>CCcc</target>
text属性将为None
"""
# 用itertext方法获取全部文本
''.join(target.itertext()) # itertext方法得到一个迭代器

# 获取上面target元素的全部文本，若碰到子节点，则获取其id属性值一起拼成一个字符串
texts = []
#获取第1段文本
if target.text:
    texts.append(target.text)
#遍历子节点
for sub in target.iterchildren():
    texts.append('-%s-' % sub.get('id', ''))
    texts.append(sub.tail)
#拼接结果
print(''.join(texts))


###### 读取./labelImg.xml ######
xml = etree.parse('labelImg.xml')
objects = xml.xpath('/annotation/object')
for a_object in objects:
    subnodes = a_object.getchildren()
    for a_subnode in subnodes:
        if a_subnode.tag == 'name':
            print(a_subnode.text)
        elif a_subnode.tag == 'bndbox':
            boundaries = a_subnode.getchildren()
            for boundary in boundaries:
                print(boundary.tag, boundary.text)