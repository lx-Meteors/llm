# 处理数据集
import ujson
import pyarrow as pa
import pyarrow.parquet as pq

def split_wiki_to_chunk_data(
        texts: list, batch_size: int=128, max_len: int=64, window_size: int=2
) -> list:
    buffer, buffer_len = [], 0
    chunk_data = []

    for i, line in enumerate(texts):
        # 计算长度
        buffer_len += len(line)
        buffer.append(line)

        if buffer_len >= batch_size or i == len(texts) - 1:
            buffer_text = "".join(buffer)

            for i in range(0, len(buffer_text), max_len - window_size):
                chunk_data.append("".join(buffer_text[i: i + max_len]))
            # 清空缓冲区
            buffer, buffer_len = [], 0
    return chunk_data







def wiki_generate(origin_file, output_file="../datasets/wiki.parquet"):
    lines = []
    with open(origin_file, "r", encoding="utf-8") as f:
        items = ujson.load(f)
        # 对每个数据添加分隔符
        for item in items:
            lines.append(item["completion"] + "<im_end>")
    chunk_data = split_wiki_to_chunk_data(lines)
    tb = pa.Table.from_arrays([pa.array(chunk_data)], names=["text"])
    pq.write_table(
        table=tb,
        where=output_file,
        row_group_size=50000,
        data_page_size=50000
    )

wiki_generate("/mnt/zhaorunsong/lx/My-LLM/data/wikipedia-cn-20230720-filtered.json")