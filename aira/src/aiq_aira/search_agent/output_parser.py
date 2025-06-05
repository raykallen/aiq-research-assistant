# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

from langchain.agents.agent import AgentOutputParser
from langchain_core.agents import AgentAction
from langchain_core.agents import AgentFinish
from langchain_core.exceptions import LangChainException
from langchain_core.utils.json import parse_json_markdown

from .prompt import SYSTEM_PROMPT

MISSING_ACTION_OR_FINAL_ANSWER_ERROR_MESSAGE = "Invalid Format: The JSON response must contain either 'action' or 'final_answer'."
MISSING_ACTION_INPUT_ERROR_MESSAGE = "Invalid Format: The JSON response for an action must contain 'action_input'."
ACTION_AND_FINAL_ANSWER_ERROR_MESSAGE = "Invalid Format: The JSON response cannot contain both 'action' and 'final_answer'."


class ReActOutputParserException(ValueError, LangChainException):

    def __init__(self,
                 observation=None,
                 missing_action_or_final_answer=False,
                 missing_action_input=False,
                 action_and_final_answer=False):
        super().__init__(observation)
        self.observation = observation
        self.missing_action_or_final_answer = missing_action_or_final_answer
        self.missing_action_input = missing_action_input
        self.action_and_final_answer = action_and_final_answer


class ReActOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that are in JSON format."""

    def get_format_instructions(self) -> str:
        return SYSTEM_PROMPT

    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            response = parse_json_markdown(text)
        except json.JSONDecodeError as exc:
            raise ReActOutputParserException(f"Could not parse LLM output: `{text}`") from exc

        has_action = "action" in response
        has_final_answer = "final_answer" in response

        if has_action and has_final_answer:
            raise ReActOutputParserException(observation=ACTION_AND_FINAL_ANSWER_ERROR_MESSAGE,
                                             action_and_final_answer=True)

        if has_action:
            if "action_input" not in response:
                raise ReActOutputParserException(observation=MISSING_ACTION_INPUT_ERROR_MESSAGE,
                                                 missing_action_input=True)

            action = response["action"]
            action_input = response["action_input"]

            return AgentAction(action, action_input, text)

        if has_final_answer:
            return AgentFinish({"output": response["final_answer"]}, text)

        raise ReActOutputParserException(observation=MISSING_ACTION_OR_FINAL_ANSWER_ERROR_MESSAGE,
                                         missing_action_or_final_answer=True)

    @property
    def _type(self) -> str:
        return "react-json-input"
