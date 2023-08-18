import re
import math


class Splitter:
    """
    helper class to split text into chunks, offers different splitting strategies
    stragies:
        - split by chunksize, starting at index 0
        - split around delimiter, with a maximum of chunksize characters around it
    """

    # splits a string into chunks of a given size in characters, only approximates the chunksize,
    # as it tries to split at the last space before the chunksize. No characters are lost, but
    # but might appear in the next chunk instead. Chunks are therefore not guaranteed to be of
    # the exact size specified, but will be smaller or equal to the specified size.
    @classmethod
    def split_by_chunksize(self, text, chunksize):
        fullLength = len(text)
        # used when end is moved to the last space before chunksize
        end = fullLength - 1

        # return single examples when batchsize == 1
        for startIndex in range(0, len(text), chunksize):
            # include characters trimmed in last iteration if trimming occured
            startIndex = min(startIndex, end)
            end = min(fullLength - 1, startIndex + chunksize)
            # move end back to last space
            while (end > startIndex and text[end] != ' '):
                end -= 1
            yield text[startIndex:end]

    # returns chunks of text evolving around a mask tokens, with text before and after the mask token
    # specified by beforeLength and afterLength. Any other mask tokens in the text are replaced with 'mask'
    # text: text to split
    # beforeLength: number of characters before the delimiter token
    # afterLength: number of characters after the delimiter token
    # delimiter: delimiter token, defaults to <mask>
    @classmethod
    def split_around_mask(self, text, beforeLength, afterLength, maskToken=r'<mask>'):
        # find all masks using regex
        maskIndices = [m.start() for m in re.finditer(maskToken, text)]
        maskLength = len(maskToken)
        for maskIndex in maskIndices:
            # get the lengths of the text before and after the mask constrained by text boundaries
            startIndex = max(0, maskIndex-beforeLength)
            endIndex = min(len(text) - 1, maskIndex + maskLength + afterLength)
            # move beforeIndex to start of word
            while (startIndex > 0 and text[startIndex] != ' '):
                startIndex -= 1
            # move afterIndex to end of word
            while (endIndex < len(text) - 1 and text[endIndex] != ' '):
                endIndex += 1
            # text before mask token with any other mask tokens removed
            before = re.sub(maskToken, 'mask', text[startIndex:maskIndex])
            # text after mask token with any other mask tokens removed
            after = re.sub(maskToken, 'mask', text[maskIndex + maskLength:endIndex])
            # yield the chunk of text
            yield before + maskToken + after

    # TODO: implement
    # takes sentence endings into account, preventing sentences from being split, which might make
    # the prediction of some masks more difficult.
    @classmethod
    def split_by_sentences(self, text):
        raise NotImplementedError

    # splits a text into chunks of a given maximum number of tokens, e.g. with max_tokens 1024
    # requires a tokenizer to determine the number of tokens in a string
    # returns a generator of chunks, each with a length of max_tokens or less
    @classmethod
    def split_by_max_tokens(self, text, tokenizer, max_tokens=512):
        # try to tokenize the full text, if it fails split it
        nb_tokens = len(tokenizer.encode(text))
        if nb_tokens <= max_tokens:
            yield text
        else:
            # split the text around the masks with smaller chunks until all chunks are small enough
            # compute the approximate factor by which the text needs to be split
            factor = math.ceil(nb_tokens / max_tokens)
            # compute number of charactes per chunk to split the text into
            chars_per_chunk = math.ceil(len(text) / factor)
            largest_chunk_tokens = math.inf
            while largest_chunk_tokens > max_tokens:
                # encode all parts and check if any are too long when using the number of characters
                # per chunk computed above (max char count divided by 2 => half before mask, half after mask)
                chunks = [*self.split_by_chunksize(text, chars_per_chunk)]
                encoded_chunks = tokenizer(chunks).input_ids
                # check if any of the chunks are too long
                largest_chunk_tokens = max([len(chunk) for chunk in encoded_chunks])
                # if any of the chunks are too long, reduce the number of characters per chunk
                # by 20% and try again
                if largest_chunk_tokens > max_tokens:
                    chars_per_chunk = math.ceil(chars_per_chunk * 0.8)
                else:
                    # yield the chunks
                    for chunk in chunks:
                        yield chunk
