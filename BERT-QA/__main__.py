from options import args
from transformers import BertForQuestionAnswering, BertTokenizer, BertConfig
from trainer.bert import BertQATrainer
from utils.displaydata import displayBertModules
from datasets import dataset_factory
import torch
import json

def train(args):

    # Load BERT QA pre-trained model
    model = BertForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_NAME)
    # high-level 顯示此模型裡的 modules
    displayBertModules(model)

    train_loader, val_loader = dataset_factory(args, tokenizer)

    BertQA = BertQATrainer(args, model, train_loader, val_loader, None)
    BertQA.train()
    pass

def test(args):

    model = BertForQuestionAnswering.from_pretrained(PRETRAINED_MODEL_NAME)
    model.load_state_dict(
        torch.load('{}/model/best_model.bin'.format(args.load_model_path))
        )
    
    test_set = dataset_factory(args, tokenizer)

    BertQA = BertQATrainer(args, model, None, None, test_set)
    BertQA.evaluate(tokenizer, _write=True)
    pass

def interaction(args):

    while True:
        qa_text = input("------------------\nPlease Enter :\n")
        
        if qa_text == 'exit':
            break

        # qa_text = '{ "sentence":"在下薩克森邦留下歷史印記的主要建築風格是文藝復興主義的一個分支[UNK][UNK]「威悉河文藝復興風格」。此外，漢諾瓦著名的海恩豪森王宮花園是歐洲巴洛克風格的典型代表。在歐斯納布魯克，人們可以找到很多古典主義和洛可可風格的建築物。這座城市的著名景點包括大教堂、威斯伐倫和約的簽署地市政廳、許多石雕和木桁架建築。下薩克森邦最大的巴洛克城堡[UNK][UNK]歐斯納布魯克城堡和最高的中世紀後哥德式建築[UNK][UNK]聖凱薩琳教堂也坐落在歐斯納布魯克。巴特伊堡的伊堡城堡和本篤會修道院在建築學和藝術史學上具有重要意義。19世紀以來，下薩克森邦造就了多位享有國際聲譽的藝術家，其中的代表性人物是畫家威廉•布施。", "question":"歐斯納布魯克有哪一座中世紀後哥德式建築是這類建築中最高的？"}'
        # qa_text = '{ "sentence":"蔡英文從小備受父母親、兄姐寵愛[26]。早期就讀臺北市私立雙連幼稚園[47]，啟蒙教育完成後，便接受國民教育[29]。1963年，就讀臺北市中山區長安國民小學[48]。1966年，四年級的她轉學到新成立的臺北市中山區吉林國民小學[48]。1971年，她以臺北市立北安國民中學第一屆畢業生畢業[48]。高級中學時，就讀臺北市立中山女子高級中學[49]，前立法院副院長、中國國民黨主席洪秀柱是大她八屆的學姐[50]。 ", "question":"誰是蔡英文總統的學姊？"}'
        # qa_test = '{ "sentence":"辛普森家庭是馬特·格朗寧為美國福斯廣播公司創作的一部成人動畫情景喜劇。該劇透過展現荷馬、美枝、霸子、花枝和奶嘴一家五口的日常生活，諷刺性地描繪了美國中產階級的生活方式。空間設定於虛構小鎮內糊的辛普森家庭，幽默地嘲諷了美國文化、社會、電視節目和人生百態。為了給製片人詹姆斯·L·布魯克斯製作一出動畫短劇，馬特·格朗寧構思出了辛普森一家人的形象。格朗寧用自己家族成員的名字逐一地給他們命名，而自己的名字則用「霸子」替代。1987年4月19日短劇成為了《特蕾西·厄爾曼秀》的一部分。在播映三季後，《辛普森家庭》得以轉正進入半小時的黃金時段，並成為了福克斯在早期達成的成功之一。", "question":"辛普森家庭是哪家公司的創作？"}'
        # qa_test = '{ "sentence":"海賊王的世界觀舞台是由世界各地的加盟國與所組成的國際組織「世界政府」所共同管理。然而，由於「海賊王」哥爾·D·羅傑被執行死刑後迎來了「大海賊時代」，結果海賊們於世界各地擴展權力，並直接與直屬世界政府的海軍作戰。本作是以島上的國家為單位，也有的島嶼只有村子、城鎮存在，大部分主要國家加入世界政府聯盟，並支持海軍討伐海賊。至於生活方式和科學技術，基本上是以現實世界海賊的「黃金時代」（17世紀到18世紀）為藍本，但是與現實世界而言還是擁有很大的差別，以作品中世界固有的獨特設定。惡魔果實服用後會依不同的果實而對應獲得不可思議的特殊能力，許多角色因其能力都擁有了超人般的戰鬥力。", "question":"在海賊王中如何得到超人般的戰鬥力？"}'

        qa_text = json.loads(qa_text)

        config = BertConfig.from_pretrained('bert-base-chinese')
        model = BertForQuestionAnswering(config)
        model.load_state_dict(
            torch.load('{}/model/best_model.bin'.format(args.load_model_path))
            )
        
        BertQA = BertQATrainer(args, model, None, None, None)
        BertQA.interaction(tokenizer, qa_text)

        pass

if __name__ == "__main__":

    # Get model tokenizer
    PRETRAINED_MODEL_NAME = args.model_name_or_path if args.model_name_or_path!=None else "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

    if(args.do_train):
        train(args)
        pass

    if(args.do_eval):
        test(args)
        pass

    if(args.do_interaction):
        interaction(args)
        pass

    pass