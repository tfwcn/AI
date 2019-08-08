
class WordHelper():
    def __init__(self):
        pass

    def create_word(self):
        alphabet=''
        for ch in range(0x4e00, 0x9fa6):
            ustr = chr(ch)
            alphabet += ustr
        for ch in range(0x3400, 0x4db6):
            ustr = chr(ch)
            alphabet += ustr
        for ch in range(0x3001, 0x3020):
            ustr = chr(ch)
            alphabet += ustr
        # 全角字母
        for ch in range(0xFF01, 0xFF66):
            ustr = chr(ch)
            alphabet += ustr
        # 日文平假名
        for ch in range(0x3041, 0x3097):
            ustr = chr(ch)
            alphabet += ustr
        # 日文片假名
        for ch in range(0x30A1, 0x30FB):
            ustr = chr(ch)
            alphabet += ustr
        # 半角
        for ch in range(0x0021, 0x007e):
            ustr = chr(ch)
            alphabet += ustr

        f2 = open('word_all.txt', 'w', encoding='utf-8')
        f2.truncate()  # 清空文件
        f2.write(alphabet)
        f2.close()
        
def main():
    word_helper = WordHelper()
    word_helper.create_word()

if __name__ == '__main__':
    main()