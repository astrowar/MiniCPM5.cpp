#include "chat_template.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

using namespace chat_template;

static const char* CASE_START = "===CASE_START===";
static const char* CASE_END   = "===CASE_END===";

static void print_case(const std::string& result) {
    std::printf("%s\n%s\n%s\n", CASE_START, result.c_str(), CASE_END);
    std::fflush(stdout);
}

int main() {
    Renderer r("<s>");

    // ---------------------------------------------------------------
    // Case 1: Simple user message (no system, no tools)
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"user", "Hello, how are you?", {}, {}},
        };
        Options opts;
        opts.add_generation_prompt = false;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 2: System + user
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"system", "You are a helpful assistant.", {}, {}},
            {"user", "What is 2+2?", {}, {}},
        };
        Options opts;
        opts.add_generation_prompt = false;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 3: User + assistant WITH reasoning (think tags in content)
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"user", "Solve 1+1", {}, {}},
            {"assistant", "Let me think... The answer is 2.", {}, {}},
        };
        // The content has no think tags, so renderer inserts empty block
        Options opts;
        opts.add_generation_prompt = false;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 4: Assistant with reasoning_content set explicitly
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"user", "Solve 2+3", {}, {}},
        };
        Message asst;
        asst.role = "assistant";
        asst.content = "The answer is 5.";
        asst.reasoning_content = "2+3 is basic arithmetic.\nI need to add them.";
        msgs.push_back(std::move(asst));

        Options opts;
        opts.add_generation_prompt = false;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 5: Generation prompt, enable_thinking = true
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"user", "What is the capital of France?", {}, {}},
        };
        Options opts;
        opts.add_generation_prompt = true;
        opts.enable_thinking = true;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 6: Generation prompt, enable_thinking = false
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"user", "What is the capital of France?", {}, {}},
        };
        Options opts;
        opts.add_generation_prompt = true;
        opts.enable_thinking = false;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 7: Generation prompt, enable_thinking = nullopt (undefined)
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs = {
            {"user", "Tell me a joke", {}, {}},
        };
        Options opts;
        opts.add_generation_prompt = true;
        // enable_thinking stays nullopt
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 8: Tool calling - assistant with tool_calls, then tool response
    // ---------------------------------------------------------------
    {
        ToolCall tc;
        tc.name = "get_weather";
        tc.arguments = {{"city", "Paris"}};

        std::vector<Message> msgs;
        msgs.push_back({"user", "What's the weather in Paris?", {}, {}});

        Message asst;
        asst.role = "assistant";
        asst.content = "";
        asst.tool_calls = {tc};
        msgs.push_back(std::move(asst));

        msgs.push_back({"tool", "{\"temp\": 22, \"condition\": \"sunny\"}", {}, {}});

        Options opts;
        opts.add_generation_prompt = true;
        opts.enable_thinking = true;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 9: Tool calling with CDATA (value contains < and newline)
    // ---------------------------------------------------------------
    {
        ToolCall tc;
        tc.name = "search_code";
        tc.arguments = {{"query", "if (a < b && c > d)\n  return true;"}};

        std::vector<Message> msgs;
        msgs.push_back({"user", "Search for comparison code", {}, {}});

        Message asst;
        asst.role = "assistant";
        asst.content = "";
        asst.tool_calls = {tc};
        msgs.push_back(std::move(asst));

        msgs.push_back({"tool", "Found 3 matches.", {}, {}});

        Options opts;
        opts.add_generation_prompt = false;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 10: System + tools + user + assistant + tool + generation
    // ---------------------------------------------------------------
    {
        std::vector<std::string> tools_json = {
            "{\"name\": \"get_weather\", \"description\": \"Get weather\", "
            "\"parameters\": {\"type\": \"object\", \"properties\": {"
            "\"city\": {\"type\": \"string\"}}, \"required\": [\"city\"]}}",
        };

        ToolCall tc;
        tc.name = "get_weather";
        tc.arguments = {{"city", "Tokyo"}};

        std::vector<Message> msgs;
        msgs.push_back({"system", "You are a weather assistant.", {}, {}});
        msgs.push_back({"user", "Weather in Tokyo?", {}, {}});

        Message asst;
        asst.role = "assistant";
        asst.content = "";
        asst.reasoning_content = "User wants Tokyo weather. I should call get_weather.";
        asst.tool_calls = {tc};
        msgs.push_back(std::move(asst));

        msgs.push_back({"tool", "{\"temp\": 15, \"condition\": \"rainy\"}", {}, {}});

        Options opts;
        opts.add_generation_prompt = true;
        opts.enable_thinking = false;
        opts.tools_json = tools_json;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 11: Multi-turn conversation with thinking
    // ---------------------------------------------------------------
    {
        std::vector<Message> msgs;
        msgs.push_back({"user", "What is AI?", {}, {}});

        Message a1;
        a1.role = "assistant";
        a1.content = "AI is the simulation of intelligence by machines.";
        a1.reasoning_content = "This is a broad question.\nI should give a concise definition.";
        msgs.push_back(std::move(a1));

        msgs.push_back({"user", "Can you elaborate?", {}, {}});

        Message a2;
        a2.role = "assistant";
        a2.content = "There are many subfields: NLP, computer vision, robotics.";
        a2.reasoning_content = "The user wants more detail.\nLet me list the main branches.";
        msgs.push_back(std::move(a2));

        Options opts;
        opts.add_generation_prompt = true;
        opts.enable_thinking = true;
        print_case(r.render(msgs, opts));
    }

    // ---------------------------------------------------------------
    // Case 12: System with tool_def_sep marker
    // ---------------------------------------------------------------
    {
        std::vector<std::string> tools_json = {
            "{\"name\": \"calc\", \"description\": \"Calculator\", "
            "\"parameters\": {\"type\": \"object\", \"properties\": {}}}"
        };

        std::vector<Message> msgs;
        msgs.push_back({"system", "You are helpful.<tool_def_sep>Additional context here.", {}, {}});
        msgs.push_back({"user", "2 * 3 = ?", {}, {}});

        Options opts;
        opts.add_generation_prompt = true;
        opts.enable_thinking = true;
        opts.tools_json = tools_json;
        print_case(r.render(msgs, opts));
    }

    return 0;
}
