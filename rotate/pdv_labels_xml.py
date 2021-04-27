from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement

"""
pdv 标注目标对象
PDVObjHead，人头
PDVObjHbody，头肩
PDVObjFbody，全身
"""

class PDVObjRect:
    # pdv 标注类别矩形的基类
    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

    def get_desc(self):
        desc = {}
        desc["cx"] = self.cx
        desc["cy"] = self.cy
        desc["w"] = self.w
        desc["h"] = self.h
        return desc

class PDVObjHead(PDVObjRect):
    # pdv 标注类别：人头，head
    def get_type_name(self):
        name = "head"
        return name

class PDVObjHbody(PDVObjRect):
    # pdv 标注类别：头肩，hbody
    def get_type_name(self):
        name = "hbody"
        return name

class PDVObjFbody():
    # pdv 标注类别：全身，fbody
    def __init__(self, cx, cy, tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y):
        self.cx = cx
        self.cy = cy
        self.tl_x = tl_x
        self.tl_y = tl_y
        self.tr_x = tr_x
        self.tr_y = tr_y
        self.br_x = br_x
        self.br_y = br_y
        self.bl_x = bl_x
        self.bl_y = bl_y

    def get_desc(self):
        desc = {}
        desc["cx"] = self.cx
        desc["cy"] = self.cy
        desc["tl_x"] = self.tl_x
        desc["tl_y"] = self.tl_y
        desc["tr_x"] = self.tr_x
        desc["tr_y"] = self.tr_y
        desc["br_x"] = self.br_x
        desc["br_y"] = self.br_y
        desc["bl_x"] = self.bl_x
        desc["bl_y"] = self.bl_y
        return desc

    def get_type_name(self):
        name = "fbody"
        return name


"""
object_data_dict
解析xml文件获取得到的标注数据，格式是：
{
    "object_name":[{"desc_0_name":desc_0_val,...,"desc_i_name":desc_i_val}],
    ...
}
"""
class PDVLabel:
    def __init__(self):
        self.root = None
        self.object_dict = {}

    def generate_xml(self, img_width, img_height, img_channels):
        """
        创建xml的root节点，必须在 add_object 和 save_xml 前调用

        [in] img_width, 图片宽
        [in] img_height, 图片高
        [in] img_channels, 图片通道数
        [out] true, 创建成功；false，创建失败
        """
        try:
            self.root = Element("annotation")
            self.root.set("verified", "yes")
            size = SubElement(self.root, "size")
            width = SubElement(size, "width")
            height = SubElement(size, "height")
            channels = SubElement(size, "channels")
            width.text = str(img_width)
            height.text = str(img_height)
            channels.text = str(img_channels)
            self.object_dict = {}
            return True
        except:
            print("generate xml failed")
            return False

    def load_xml(self, xml_file_path):
        """
        加载一个xml，更新root

        [in] xml_file_path, 加载xml文件路径
        [out] true, 创建成功；false，创建失败
        """
        root = None
        object_dict = {}
        try:
            xmltree = ElementTree.parse(xml_file_path)
            root = xmltree.getroot()

            # width = int(xmltree.find("size").find("width").text)
            # height = int(xmltree.find("size").find("height").text)
            # channels = int(xmltree.find("size").find("channels").text)

            for object_iter in xmltree.findall("object"):
                obj_type = object_iter.find("type").text
                obj_name = object_iter.find("name").text
                obj_desc = object_iter.find("description")
                if obj_type == "box":
                    if obj_name not in object_dict:
                        object_dict[obj_name] = []
                    box_desc = {}
                    box_desc["cx"] = obj_desc.find("cx").text
                    box_desc["cy"] = obj_desc.find("cy").text
                    box_desc["w"] = obj_desc.find("w").text
                    box_desc["h"] = obj_desc.find("h").text
                    object_dict[obj_name].append(box_desc)
                elif obj_type == "robox_pts":
                    if obj_name not in object_dict:
                        object_dict[obj_name] = []
                    box_desc = {}
                    box_desc["cx"] = obj_desc.find("cx").text
                    box_desc["cy"] = obj_desc.find("cy").text
                    box_desc["tlx"] = obj_desc.find("tlx").text
                    box_desc["tly"] = obj_desc.find("tly").text
                    box_desc["trx"] = obj_desc.find("trx").text
                    box_desc["try"] = obj_desc.find("try").text
                    box_desc["brx"] = obj_desc.find("brx").text
                    box_desc["bry"] = obj_desc.find("bry").text
                    box_desc["blx"] = obj_desc.find("blx").text
                    box_desc["bly"] = obj_desc.find("bly").text
                    object_dict[obj_name].append(box_desc)
                else:
                    print("unknown object type [{}]".format(obj_type))
        except:
            return False

        self.root = root
        self.object_dict = object_dict
        return True

    def get_objects(self):
        """
        获取解析得到标注目标数据

        [in] 无
        [out] object_dict，标注数据，是一个字典，格式参照[object_data_dict]
        """
        return self.object_dict

    def add_object(self, pdv_obj):
        """
        添加一个标注目标节点

        [in] pdv_obj，标注目标对象，参照[pdv 标注目标对象]
        [out] true, 添加成功；false，添加失败
        """
        if self.root is None:
            print("add object failed, you must use generate_xml first")
            return False
        if pdv_obj is None:
            print("pdv_obj is none, add object failed")
            return False
        pdv_obj_desc = pdv_obj.get_desc()
        pdv_obj_type = pdv_obj.get_type_name()
        if pdv_obj_type == "head":
            e_obj = SubElement(self.root, "object")
            e_obj_type = SubElement(e_obj, "type")
            e_obj_type.text = "box"
            e_obj_name = SubElement(e_obj, "name")
            e_obj_name.text = "head"
            e_obj_desc = SubElement(e_obj, "description")
            e_desc_cx = SubElement(e_obj_desc, "cx")
            e_desc_cy = SubElement(e_obj_desc, "cy")
            e_desc_w = SubElement(e_obj_desc, "w")
            e_desc_h = SubElement(e_obj_desc, "h")
            e_desc_cx.text = str(pdv_obj_desc["cx"])
            e_desc_cy.text = str(pdv_obj_desc["cy"])
            e_desc_w.text = str(pdv_obj_desc["w"])
            e_desc_h.text = str(pdv_obj_desc["h"])

            if pdv_obj_type not in self.object_dict:
                self.object_dict[pdv_obj_type] = []
            box_desc = {}
            box_desc["cx"] = e_desc_cx.text
            box_desc["cy"] = e_desc_cy.text
            box_desc["w"] = e_desc_w.text
            box_desc["h"] = e_desc_h.text
            self.object_dict[pdv_obj_type].append(box_desc)
        elif pdv_obj_type == "hbody":
            e_obj = SubElement(self.root, "object")
            e_obj_type = SubElement(e_obj, "type")
            e_obj_type.text = "box"
            e_obj_name = SubElement(e_obj, "name")
            e_obj_name.text = "hbody"
            e_obj_desc = SubElement(e_obj, "description")
            e_desc_cx = SubElement(e_obj_desc, "cx")
            e_desc_cy = SubElement(e_obj_desc, "cy")
            e_desc_w = SubElement(e_obj_desc, "w")
            e_desc_h = SubElement(e_obj_desc, "h")
            e_desc_cx.text = str(pdv_obj_desc["cx"])
            e_desc_cy.text = str(pdv_obj_desc["cy"])
            e_desc_w.text = str(pdv_obj_desc["w"])
            e_desc_h.text = str(pdv_obj_desc["h"])
            if pdv_obj_type not in self.object_dict:
                self.object_dict[pdv_obj_type] = []
            box_desc = {}
            box_desc["cx"] = e_desc_cx.text
            box_desc["cy"] = e_desc_cy.text
            box_desc["w"] = e_desc_w.text
            box_desc["h"] = e_desc_h.text
            self.object_dict[pdv_obj_type].append(box_desc)
        elif pdv_obj_type == "fbody":
            e_obj = SubElement(self.root, "object")
            e_obj_type = SubElement(e_obj, "type")
            e_obj_type.text = "robox_pts"
            e_obj_name = SubElement(e_obj, "name")
            e_obj_name.text = "fbody"
            e_obj_desc = SubElement(e_obj, "description")
            e_desc_cx = SubElement(e_obj_desc, "cx")
            e_desc_cy = SubElement(e_obj_desc, "cy")
            e_desc_tlx = SubElement(e_obj_desc, "tlx")
            e_desc_tly = SubElement(e_obj_desc, "tly")
            e_desc_trx = SubElement(e_obj_desc, "trx")
            e_desc_try = SubElement(e_obj_desc, "try")
            e_desc_brx = SubElement(e_obj_desc, "brx")
            e_desc_bry = SubElement(e_obj_desc, "bry")
            e_desc_blx = SubElement(e_obj_desc, "blx")
            e_desc_bly = SubElement(e_obj_desc, "bly")
            e_desc_cx.text = str(pdv_obj_desc["cx"])
            e_desc_cy.text = str(pdv_obj_desc["cy"])
            e_desc_tlx.text = str(pdv_obj_desc["tl_x"])
            e_desc_tly.text = str(pdv_obj_desc["tl_y"])
            e_desc_trx.text = str(pdv_obj_desc["tr_x"])
            e_desc_try.text = str(pdv_obj_desc["tr_y"])
            e_desc_brx.text = str(pdv_obj_desc["br_x"])
            e_desc_bry.text = str(pdv_obj_desc["br_y"])
            e_desc_blx.text = str(pdv_obj_desc["bl_x"])
            e_desc_bly.text = str(pdv_obj_desc["bl_y"])

            if pdv_obj_type not in self.object_dict:
                self.object_dict[pdv_obj_type] = []
            box_desc = {}
            box_desc["cx"] = e_desc_cx.text
            box_desc["cy"] = e_desc_cy.text
            box_desc["tlx"] = e_desc_tlx.text
            box_desc["tly"] = e_desc_tly.text
            box_desc["trx"] = e_desc_trx.text
            box_desc["try"] = e_desc_try.text
            box_desc["brx"] = e_desc_brx.text
            box_desc["bry"] = e_desc_bry.text
            box_desc["blx"] = e_desc_blx.text
            box_desc["bly"] = e_desc_bly.text
            self.object_dict[pdv_obj_type].append(box_desc)
        else:
            print("unknown object type, add object failed")
            return False
        return True

    def indent(self, elem, level=0):
        """
        对节点元素添加缩进

        [in,out] elem，元素节点
        [in] level, 层级
        """
        i = "\n" + level*"    "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "    "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self.indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def save_xml(self, save_file_path):
        """
        保存当前xml到指定路径

        [in] save_file_path, 保存文件路径
        [out] true, 添加成功；false，添加失败
        """
        if self.root is None:
            print("root is none, save file failed")
            return False
        self.indent(self.root)
        xml_tree = ElementTree.ElementTree(self.root)
        xml_tree.write(save_file_path, encoding="utf-8", xml_declaration=True)
        return True


"""
单元测试
1.创建一个pdv标注格式的xml文件，写入到当前目录的test.xml
2.读取一个pdv标注格式的xml文件（test.xml）, 打印标注目标信息；添加标注目标，并写入当前目录的test_new.xml
3.读取一个pdv标注格式的xml文件（test_new.xml）, 打印标注目标信息
"""
if __name__ == "__main__":
    pdv_label = PDVLabel()
    pdv_label.generate_xml(1920, 1080, 3)
    pdv_obj_head = PDVObjHead(0.1, 0.1, 0.2, 0.2)
    pdv_obj_hbody = PDVObjHbody(1, 1, 2, 2)
    pdv_obj_fbody = PDVObjFbody(1, 1, 2, 2, 3, 3, 4, 4, 5, 5)
    pdv_label.add_object(pdv_obj_head)
    pdv_label.add_object(pdv_obj_hbody)
    pdv_label.add_object(pdv_obj_fbody)
    pdv_label.save_xml("./test.xml")

    pdv_label_2 = PDVLabel()
    pdv_label_2.load_xml("./test.xml")
    objects_dict = pdv_label.get_objects()
    print(objects_dict)
    pdv_obj_fbody_2 = PDVObjFbody(10, 10, 20, 20, 30, 30, 40, 40, 50, 50)
    pdv_label_2.add_object(pdv_obj_fbody_2)
    pdv_label_2.save_xml("./test_new.xml")

    pdv_label.load_xml("./test_new.xml")
    objects_dict = pdv_label.get_objects()
    print(objects_dict)
