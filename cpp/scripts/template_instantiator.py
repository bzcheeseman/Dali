# from the land of Szymon comes this wonderful IDE
# use gracefully


"""
EXAMPLE:

// BEGIN TEMPLATE SPECIALIZATIONS
// TEMPLATE
// [float, double] masked_cross_entropy
// [std::shared_ptr<Mat<RETURN_TYPE>>]
// [uint&]
// [const eigen_index_block&, const eigen_index_vector&, shared_eigen_index_vector, uint, int, const uint&, const int&]
// [const eigen_index_block&, const eigen_index_vector&, shared_eigen_index_vector, uint, int, const uint&, const int&]
// [const eigen_index_block, const eigen_index_block_scalar]
//
// TEMPLATE
// [float, double] masked_sum
// [std::shared_ptr<Mat<RETURN_TYPE>>]
// [uint&]
// [const eigen_index_block&, const eigen_index_vector&, shared_eigen_index_vector, uint, int, const uint&, const int&]
// [const eigen_index_block&, const eigen_index_vector&, shared_eigen_index_vector, uint, int, const uint&, const int&]
// [RETURN_TYPE]
//
// END TEMPLATE SPECIALIZATIONS
"""
import hashlib
import sys

from itertools import product


USAGE = """Usage:
%s FILE"""

def generate_combinations(return_types, function_name, arguments_lists):
    for arrangement in product(return_types, *arguments_lists):
        return_type, arguments = arrangement[0], ', '.join(arrangement[1:])

        line = 'template RETURN_TYPE %s(%s);' % (function_name, arguments)
        yield line.replace("RETURN_TYPE", return_type)

def parse_array(word):
    content = word[1:-1]
    return [c.strip() for c in content.split(',')]

def generate_templates(lines):
    words = ' '.join(lines).split(' ')
    words = [word.strip() for word in words]

    new_words = []
    args_on = False
    args = []
    for word in words:
        if word[0] == '[':
            args_on = True

        if args_on:
            args.append(word)
        else:
            new_words.append(word)
        if word[-1] == ']':
            args_on = False
            new_words.append(' '.join(args))
            args = []
    words = new_words
    output_lines = []

    return_types = parse_array(words[0])
    function_name = words[1]
    arguments_lists = []
    for word in words[2:]:
        arguments_lists.append(parse_array(word))
    return generate_combinations(return_types, function_name, arguments_lists)



if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(USAGE % (sys.argv[0],))
    source = sys.argv[1]
    template_lines = []

    reading_on = False
    output_lines = []
    with open(source, "rt") as f:
        for line in f:
            if line.startswith(r"//"):
                line_content = line[2:].strip()
                if line_content == "BEGIN TEMPLATE SPECIALIZATIONS":
                    reading_on = True
                elif line_content == "END TEMPLATE SPECIALIZATIONS":
                    if len(template_lines) > 0:
                        output_lines.extend(generate_templates(template_lines))
                        template_lines = []
                    reading_on = False
                elif reading_on:
                    if line_content == 'TEMPLATE':
                        if len(template_lines) > 0:
                            output_lines.extend(generate_templates(template_lines))
                            template_lines = []
                    else:
                       template_lines.append(line_content)
                else:
                    output_lines.append(line[:-1])
            else:
                if not reading_on:
                    output_lines.append(line[:-1])

    output_file = '.'.join(source.split('.')[:-1])
    output_content = '\n'.join(output_lines)
    output_content_hash = hashlib.md5(output_content.encode('utf-8')).hexdigest()
    output_file_hash = None
    with open(output_file, "rt") as f:
        output_file_hash = hashlib.md5(f.read().encode('utf-8')).hexdigest()

    # only update file if contents changed
    if output_file_hash != output_content_hash:
        print("Reinstantiated template for %s" % (source,))
        with open(output_file, "wt") as f:
            f.write(output_content)
    else:
        print("Template instantiation not needed for %s" % (source,))
